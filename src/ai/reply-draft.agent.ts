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
 * API KEY ROTATION
 * ----------------
 * Set an expiration date on the key in the OpenAI Platform (Platform →
 * API keys → key options → Expiration) and mirror that date in the
 * environment variable OPENAI_KEY_EXPIRES_AT (ISO-8601, e.g.
 * "2025-03-01T00:00:00Z").  The startup check below will throw if the key
 * has expired and warn 7 days ahead of expiry, giving on-call enough notice
 * to rotate before the key stops working.
 */

// ---------------------------------------------------------------------------
// API-key lifetime guard (Finding 3)
// ---------------------------------------------------------------------------

const KEY_EXPIRY_WARNING_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

(function assertKeyNotExpired() {
  const raw = process.env.OPENAI_KEY_EXPIRES_AT;
  if (!raw) {
    // No expiry configured: emit a one-time warning so operators know they
    // should set one in both OpenAI Platform settings and the environment.
    console.warn(
      '[reply-draft] OPENAI_KEY_EXPIRES_AT is not set. ' +
        'Set an expiration date on the API key in the OpenAI Platform and ' +
        'mirror it here so key expiry is enforced at startup.',
    );
    return;
  }

  const expiresAt = new Date(raw);
  if (isNaN(expiresAt.getTime())) {
    throw new Error(
      `[reply-draft] OPENAI_KEY_EXPIRES_AT value "${raw}" is not a valid ISO-8601 date.`,
    );
  }

  const now = Date.now();
  if (now >= expiresAt.getTime()) {
    throw new Error(
      `[reply-draft] OPENAI_API_KEY expired at ${expiresAt.toISOString()}. ` +
        'Rotate the key in the OpenAI Platform and update OPENAI_API_KEY and ' +
        'OPENAI_KEY_EXPIRES_AT before restarting.',
    );
  }

  if (expiresAt.getTime() - now <= KEY_EXPIRY_WARNING_MS) {
    console.warn(
      `[reply-draft] OPENAI_API_KEY expires on ${expiresAt.toISOString()} ` +
        '(within 7 days). Rotate the key now to avoid service interruption.',
    );
  }
})();

// ---------------------------------------------------------------------------
// Model configuration (Finding 2)
// ---------------------------------------------------------------------------

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-4o-mini';

/**
 * Models that do NOT accept a custom `temperature` or `top_p`.
 * Remove the model name from this set if OpenAI re-enables the parameter.
 */
const MODELS_WITHOUT_TEMPERATURE = new Set(['gpt-6-astra']);

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

// ---------------------------------------------------------------------------
// Retry / back-off helper (Finding 1)
// ---------------------------------------------------------------------------

const MAX_ATTEMPTS = 4;
const BASE_BACKOFF_MS = 500;

/**
 * Returns the number of milliseconds to wait before the next attempt.
 *
 * Prefers the `Retry-After` header value (in seconds) when present;
 * otherwise uses truncated exponential back-off with full jitter.
 */
function retryDelayMs(attempt: number, retryAfterSeconds?: number): number {
  if (retryAfterSeconds != null && retryAfterSeconds > 0) {
    return retryAfterSeconds * 1000;
  }
  // Exponential back-off: BASE * 2^attempt, jittered ±25 %.
  const base = BASE_BACKOFF_MS * Math.pow(2, attempt);
  const jitter = base * 0.25 * (Math.random() * 2 - 1);
  return Math.round(base + jitter);
}

function isRetryableError(err: unknown): { retryable: boolean; retryAfterSeconds?: number } {
  if (err && typeof err === 'object') {
    const status: unknown = (err as Record<string, unknown>).status ??
      (err as Record<string, unknown>).statusCode;
    if (status === 429 || status === 503) {
      // Try to read a Retry-After header buried in the error object.
      const headers: unknown = (err as Record<string, unknown>).responseHeaders ??
        (err as Record<string, unknown>).headers;
      let retryAfterSeconds: number | undefined;
      if (headers && typeof headers === 'object') {
        const ra = (headers as Record<string, unknown>)['retry-after'];
        if (ra != null) {
          const parsed = Number(ra);
          if (!isNaN(parsed)) {
            retryAfterSeconds = parsed;
          }
        }
      }
      return { retryable: true, retryAfterSeconds };
    }
  }
  return { retryable: false };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  // Gate temperature on model capability (Finding 2).
  const supportsTemperature = !MODELS_WITHOUT_TEMPERATURE.has(REPLY_DRAFT_MODEL);

  let lastError: unknown;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      const { text } = await generateText({
        model: openai(REPLY_DRAFT_MODEL),
        system: REPLY_DRAFT_SYSTEM_PROMPT,
        prompt: buildReplyDraftPrompt(input),
        // Support replies should read consistently between agents.
        // Omitted automatically for models that reject the parameter (e.g. gpt-6-astra).
        ...(supportsTemperature ? { temperature: 0.3 } : {}),
        maxOutputTokens: 500,
      });

      const body = text.trim();
      if (!body) {
        throw new Error('Reply draft came back empty');
      }

      return { model: REPLY_DRAFT_MODEL, body };
    } catch (err: unknown) {
      lastError = err;

      const { retryable, retryAfterSeconds } = isRetryableError(err);
      const isLastAttempt = attempt === MAX_ATTEMPTS - 1;

      if (!retryable || isLastAttempt) {
        // Non-transient error, or we've exhausted all attempts — give up.
        throw err;
      }

      const delay = retryDelayMs(attempt, retryAfterSeconds);
      console.warn(
        `[reply-draft] Transient error (status ${(err as Record<string, unknown>).status ?? (err as Record<string, unknown>).statusCode}), ` +
          `retrying in ${delay} ms (attempt ${attempt + 1}/${MAX_ATTEMPTS}).`,
      );
      await sleep(delay);
    }
  }

  // Should be unreachable, but TypeScript needs a guaranteed throw.
  throw lastError;
}
