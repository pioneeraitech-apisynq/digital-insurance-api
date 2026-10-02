import { createOpenAI } from '@ai-sdk/openai';
import { generateText } from 'ai';
import {
  REPLY_DRAFT_SYSTEM_PROMPT,
  ReplyDraftInput,
  buildReplyDraftPrompt,
} from './prompts/reply-draft.prompt';

/**
 * Reply draft helper.
 *
 * Runs OpenAI gpt-4o-mini through the Vercel AI SDK. Support agents get a
 * first draft of a reply to a customer message; a human always edits and sends
 * it, so nothing here is customer-facing on its own.
 *
 * The base URL is read from OPENAI_BASE_URL rather than left at the SDK
 * default, so the same build can be pointed at the APISynQ gateway (which
 * records the call and applies the data-class policy) instead of talking to
 * api.openai.com directly.
 *
 * ─── KEY ROTATION (operator action required) ──────────────────────────────
 * The OpenAI knowledge pack requires:
 *   1. Set an expiration date on OPENAI_API_KEY via the OpenAI dashboard
 *      (platform.openai.com → API keys → key settings → "Expiration").
 *   2. Enforce a maximum key lifetime at the org/project level
 *      (platform.openai.com → Organisation settings → "Maximum key lifetime").
 *   3. Rotate the key on a schedule (recommended: ≤ 90 days).
 *   4. After creating a new key, set OPENAI_API_KEY_CREATED_AT (ISO-8601 date)
 *      in the deployment environment so the runtime guard below can warn when
 *      the key is approaching its maximum age.
 * ──────────────────────────────────────────────────────────────────────────
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-4o-mini';

// ---------------------------------------------------------------------------
// Retry / back-off configuration
// ---------------------------------------------------------------------------

/** HTTP status codes that warrant a retry with back-off. */
const RETRYABLE_STATUSES = new Set([429, 503]);

/** Maximum number of attempts (1 initial + N-1 retries). */
const MAX_ATTEMPTS = 4;

/** Base delay in ms for exponential back-off when Retry-After is absent. */
const BASE_BACKOFF_MS = 500;

/** Cap on how long we will ever wait between retries (ms). */
const MAX_BACKOFF_MS = 30_000;

// ---------------------------------------------------------------------------
// Key-age governance guard
// ---------------------------------------------------------------------------

/**
 * Maximum recommended key age in days. Matches the rotation schedule called
 * out in the operator instructions above. Override via MAX_KEY_AGE_DAYS env
 * var if your organisation enforces a different maximum.
 */
const MAX_KEY_AGE_DAYS = Number(process.env.MAX_KEY_AGE_DAYS ?? 90);

/**
 * Warn at startup if OPENAI_API_KEY_CREATED_AT indicates the active key is
 * older than MAX_KEY_AGE_DAYS. This is a belt-and-suspenders runtime signal;
 * the primary enforcement must be configured in the OpenAI dashboard (see
 * operator instructions above).
 */
function warnIfKeyLooksStale(): void {
  const createdAt = process.env.OPENAI_API_KEY_CREATED_AT;
  if (!createdAt) {
    // Env var not set — we cannot check; silently skip.
    return;
  }

  const created = new Date(createdAt);
  if (isNaN(created.getTime())) {
    console.warn(
      '[reply-draft] OPENAI_API_KEY_CREATED_AT is not a valid ISO-8601 date' +
        ' — skipping key-age check.',
    );
    return;
  }

  const ageMs = Date.now() - created.getTime();
  const ageDays = ageMs / (1000 * 60 * 60 * 24);
  if (ageDays > MAX_KEY_AGE_DAYS) {
    console.warn(
      `[reply-draft] OPENAI_API_KEY is ${Math.floor(ageDays)} days old` +
        ` (max recommended: ${MAX_KEY_AGE_DAYS} days).` +
        ' Rotate the key and update the OpenAI dashboard expiration setting.',
    );
  }
}

warnIfKeyLooksStale();

// ---------------------------------------------------------------------------
// OpenAI client
// ---------------------------------------------------------------------------

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ReplyDraft {
  model: string;
  body: string;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Extract an HTTP status code from whatever the AI SDK throws.
 * The SDK may surface it as `error.status`, `error.statusCode`, or nested on
 * `error.cause` (the raw Response / fetch error).
 */
function getStatusCode(error: unknown): number | undefined {
  if (error == null || typeof error !== 'object') return undefined;
  const e = error as Record<string, unknown>;
  if (typeof e['status'] === 'number') return e['status'];
  if (typeof e['statusCode'] === 'number') return e['statusCode'];
  // Some SDK versions attach the raw Response as `cause`.
  const cause = e['cause'];
  if (cause != null && typeof cause === 'object') {
    const c = cause as Record<string, unknown>;
    if (typeof c['status'] === 'number') return c['status'];
  }
  return undefined;
}

/**
 * Extract the value of the `Retry-After` header (seconds) from whatever the
 * AI SDK throws, or return undefined when it is not available.
 */
function getRetryAfterSeconds(error: unknown): number | undefined {
  if (error == null || typeof error !== 'object') return undefined;
  const e = error as Record<string, unknown>;

  // Direct property some SDK versions expose.
  if (typeof e['retryAfter'] === 'number') return e['retryAfter'];

  // Raw Response object attached as `cause` or `response`.
  for (const key of ['cause', 'response']) {
    const candidate = e[key];
    if (candidate != null && typeof candidate === 'object') {
      const c = candidate as Record<string, unknown>;
      // fetch Response exposes `.headers` with a `.get()` method.
      if (typeof (c['headers'] as any)?.get === 'function') {
        const val = (c['headers'] as Headers).get('retry-after');
        if (val !== null) {
          const seconds = Number(val);
          if (!isNaN(seconds)) return seconds;
        }
      }
    }
  }

  return undefined;
}

/**
 * Compute the delay (ms) before the next retry attempt.
 *
 * Prefers the `Retry-After` header value supplied by the API. Falls back to
 * exponential back-off with full jitter when the header is absent.
 */
function retryDelayMs(error: unknown, attempt: number): number {
  const retryAfterSecs = getRetryAfterSeconds(error);
  if (retryAfterSecs !== undefined && retryAfterSecs > 0) {
    // Add a small jitter (≤ 200 ms) so multiple instances do not all fire at
    // exactly the same moment after a rate-limit window resets.
    return retryAfterSecs * 1000 + Math.random() * 200;
  }

  // Exponential back-off: BASE * 2^attempt, capped at MAX_BACKOFF_MS, with
  // full jitter (uniform [0, computed_cap]).
  const exponential = Math.min(BASE_BACKOFF_MS * 2 ** attempt, MAX_BACKOFF_MS);
  return Math.random() * exponential;
}

/** Tiny sleep utility. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  let lastError: unknown;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      const { text } = await generateText({
        model: openai(REPLY_DRAFT_MODEL),
        system: REPLY_DRAFT_SYSTEM_PROMPT,
        prompt: buildReplyDraftPrompt(input),
        // Support replies should read consistently between agents.
        temperature: 0.3,
        maxOutputTokens: 500,
      });

      const body = text.trim();
      if (!body) {
        throw new Error('Reply draft came back empty');
      }

      return { model: REPLY_DRAFT_MODEL, body };
    } catch (error: unknown) {
      lastError = error;

      const status = getStatusCode(error);
      const isRetryable = status !== undefined && RETRYABLE_STATUSES.has(status);
      const isLastAttempt = attempt === MAX_ATTEMPTS - 1;

      if (!isRetryable || isLastAttempt) {
        // Non-retryable error, or we have exhausted all attempts.
        throw error;
      }

      const delay = retryDelayMs(error, attempt);
      console.warn(
        `[reply-draft] OpenAI responded with HTTP ${status} on attempt` +
          ` ${attempt + 1}/${MAX_ATTEMPTS}. Retrying in ${Math.round(delay)} ms.`,
      );
      await sleep(delay);
    }
  }

  // Should be unreachable, but TypeScript needs the explicit throw.
  throw lastError;
}
