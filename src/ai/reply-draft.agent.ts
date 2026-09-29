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
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-4o-mini';

/**
 * HTTP status codes that are transient and safe to retry.
 * 429 – rate-limited (slow_down); 503 – server overloaded.
 */
const RETRYABLE_STATUS_CODES = new Set([429, 503]);

/** Maximum number of attempts (1 initial + 3 retries). */
const MAX_ATTEMPTS = 4;

/** Base delay for exponential backoff in milliseconds. */
const BASE_DELAY_MS = 1_000;

/** Upper cap on any single wait to avoid unbounded hangs (30 s). */
const MAX_DELAY_MS = 30_000;

/**
 * Returns how long to wait (ms) before the next attempt.
 *
 * Prefers the `Retry-After` header value (integer seconds or HTTP-date) when
 * the API includes one; otherwise falls back to jittered exponential backoff.
 */
function resolveDelay(error: unknown, attempt: number): number {
  // The Vercel AI SDK surfaces response headers as `responseHeaders` on the
  // thrown error object (a plain Record<string, string>).
  const headers =
    error != null &&
    typeof error === 'object' &&
    'responseHeaders' in error &&
    typeof (error as Record<string, unknown>).responseHeaders === 'object'
      ? ((error as Record<string, unknown>).responseHeaders as Record<
          string,
          string
        >)
      : null;

  if (headers) {
    const raw =
      headers['retry-after'] ?? headers['Retry-After'] ?? headers['x-ratelimit-reset-requests'];

    if (raw != null) {
      // Retry-After can be an integer (seconds) or an HTTP-date string.
      const seconds = Number(raw);
      if (!Number.isNaN(seconds) && seconds >= 0) {
        // Clamp to MAX_DELAY_MS so a rogue gateway can't stall us forever.
        return Math.min(seconds * 1_000, MAX_DELAY_MS);
      }

      // HTTP-date format: "Wed, 21 Oct 2015 07:28:00 GMT"
      const retryAt = Date.parse(raw);
      if (!Number.isNaN(retryAt)) {
        const delta = retryAt - Date.now();
        return Math.min(Math.max(delta, 0), MAX_DELAY_MS);
      }
    }
  }

  // Exponential backoff with full jitter: random value in [0, BASE * 2^attempt).
  const ceiling = Math.min(BASE_DELAY_MS * 2 ** attempt, MAX_DELAY_MS);
  return Math.random() * ceiling;
}

/** Returns true when the error represents a transient, retryable condition. */
function isRetryable(error: unknown): boolean {
  if (error == null || typeof error !== 'object') return false;
  const status =
    (error as Record<string, unknown>).statusCode ??
    (error as Record<string, unknown>).status;
  return typeof status === 'number' && RETRYABLE_STATUS_CODES.has(status);
}

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

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
      // Never retry non-transient errors (auth failures, bad requests, etc.)
      // or the sentinel we throw ourselves for an empty body.
      if (!isRetryable(err)) {
        throw err;
      }

      lastError = err;

      if (attempt < MAX_ATTEMPTS - 1) {
        const delay = resolveDelay(err, attempt);
        await new Promise<void>((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  // All attempts exhausted – re-throw the last transient error.
  throw lastError;
}
