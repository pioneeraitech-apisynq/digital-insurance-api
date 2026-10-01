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
 * API key governance (production requirement):
 *   OPENAI_API_KEY MUST be a service-account project key (not a personal user
 *   key) with a configured maximum lifetime set in the OpenAI Platform →
 *   Project → API keys settings. The organisation-level "maximum key lifetime"
 *   control should also be enabled so that no project key can be created
 *   without an expiry date. Keys must be rotated before expiry; document the
 *   rotation schedule in your team runbook and in the OpenAI Platform settings
 *   description field for this key.
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-4o-mini';

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

// ---------------------------------------------------------------------------
// Retry configuration
// ---------------------------------------------------------------------------

/** Maximum number of attempts (1 initial + N-1 retries). */
const MAX_ATTEMPTS = 4;

/**
 * Base delay for exponential backoff when no Retry-After header is present.
 * Attempt k (0-indexed) waits BASE_DELAY_MS * 2^k, capped at MAX_DELAY_MS.
 */
const BASE_DELAY_MS = 500;
const MAX_DELAY_MS = 16_000;

/** HTTP status codes that warrant a retry with backoff. */
const RETRYABLE_STATUSES = new Set([429, 503]);

/** Sleep for `ms` milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Extract a wait duration (ms) from the response that triggered a retryable
 * error.  The Vercel AI SDK surfaces rate-limit responses as an `APICallError`
 * which carries `statusCode` and `responseHeaders`.
 *
 * Falls back to capped exponential backoff when the header is absent or
 * cannot be parsed.
 */
function resolveDelayMs(err: unknown, attempt: number): number {
  const exponentialMs = Math.min(
    BASE_DELAY_MS * Math.pow(2, attempt),
    MAX_DELAY_MS,
  );

  if (err !== null && typeof err === 'object') {
    // The Vercel AI SDK wraps HTTP errors as AI_APICallError; the header map
    // may be a plain object or a Headers instance — handle both.
    const headers: unknown =
      (err as Record<string, unknown>).responseHeaders ?? null;

    let retryAfterValue: string | null = null;

    if (headers instanceof Headers) {
      retryAfterValue = headers.get('retry-after');
    } else if (headers !== null && typeof headers === 'object') {
      const raw = (headers as Record<string, unknown>)['retry-after'];
      if (typeof raw === 'string') {
        retryAfterValue = raw;
      }
    }

    if (retryAfterValue !== null) {
      const seconds = parseFloat(retryAfterValue);
      if (!Number.isNaN(seconds) && seconds >= 0) {
        // Clamp to MAX_DELAY_MS so a misbehaving header can't stall the worker
        // for an unreasonably long time.
        return Math.min(seconds * 1_000, MAX_DELAY_MS);
      }
    }
  }

  return exponentialMs;
}

/** Returns true when the error represents a 429 or 503 response. */
function isRetryable(err: unknown): boolean {
  if (err !== null && typeof err === 'object') {
    const status = (err as Record<string, unknown>).statusCode;
    if (typeof status === 'number' && RETRYABLE_STATUSES.has(status)) {
      return true;
    }
  }
  return false;
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
    } catch (err: unknown) {
      lastError = err;

      // Non-retryable errors (4xx other than 429, network errors that are not
      // transient overloads, empty-draft, etc.) bubble up immediately.
      if (!isRetryable(err)) {
        throw err;
      }

      const isLastAttempt = attempt === MAX_ATTEMPTS - 1;
      if (isLastAttempt) {
        break;
      }

      const delayMs = resolveDelayMs(err, attempt);
      await sleep(delayMs);
    }
  }

  // All attempts exhausted on retryable errors.
  throw lastError;
}
