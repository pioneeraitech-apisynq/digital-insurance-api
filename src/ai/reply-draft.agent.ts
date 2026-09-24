import { createOpenAI } from '@ai-sdk/openai';
import { APICallError } from '@ai-sdk/provider';
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

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

// ---------------------------------------------------------------------------
// Retry / backoff helpers
// ---------------------------------------------------------------------------

/** HTTP status codes that are transient and must be retried. */
const RETRIABLE_STATUS_CODES = new Set([429, 503]);

const RETRY_MAX_ATTEMPTS = 4;   // 1 original + 3 retries
const RETRY_BASE_DELAY_MS = 500;
const RETRY_MAX_DELAY_MS  = 30_000;

/**
 * Resolves the number of milliseconds to wait before the next attempt.
 *
 * Priority order:
 *  1. `Retry-After: <seconds>` response header supplied by the API.
 *  2. Capped exponential backoff with full jitter:
 *     `random(0, min(maxDelay, base * 2^attempt))`.
 */
function resolveRetryDelay(attempt: number, retryAfterHeader?: string): number {
  if (retryAfterHeader) {
    const seconds = parseFloat(retryAfterHeader);
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.ceil(seconds * 1_000);
    }
  }

  const exponentialCap = Math.min(
    RETRY_MAX_DELAY_MS,
    RETRY_BASE_DELAY_MS * Math.pow(2, attempt),
  );
  // Full jitter: uniform random in [0, cap]
  return Math.floor(Math.random() * (exponentialCap + 1));
}

/** Returns true when an error represents a transient API failure worth retrying. */
function isRetriable(err: unknown): err is APICallError {
  return (
    APICallError.isInstance(err) &&
    RETRIABLE_STATUS_CODES.has((err as APICallError).statusCode)
  );
}

/**
 * Runs `fn`, retrying on 429 / 503 responses up to `RETRY_MAX_ATTEMPTS`
 * total attempts.  The `Retry-After` header is honoured when present;
 * otherwise capped exponential backoff with full jitter is used.
 */
async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < RETRY_MAX_ATTEMPTS; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (!isRetriable(err) || attempt === RETRY_MAX_ATTEMPTS - 1) {
        throw err;
      }

      lastError = err;
      const retryAfter = (err as APICallError).responseHeaders?.['retry-after'];
      const delayMs = resolveRetryDelay(attempt, retryAfter);

      await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
    }
  }

  // Unreachable, but satisfies the TypeScript control-flow checker.
  throw lastError;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await withRetry(() =>
    generateText({
      model: openai(REPLY_DRAFT_MODEL),
      system: REPLY_DRAFT_SYSTEM_PROMPT,
      prompt: buildReplyDraftPrompt(input),
      // Support replies should read consistently between agents.
      temperature: 0.3,
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
