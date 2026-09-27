import { createOpenAI } from '@ai-sdk/openai';
import { APICallError, generateText } from 'ai';
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

/** Status codes that are transient and safe to retry. */
const RETRYABLE_STATUS_CODES = new Set([429, 503]);

/** Maximum number of attempts (1 initial + N-1 retries). */
const MAX_ATTEMPTS = 4;

/** Cap applied to the computed backoff delay (ms). */
const MAX_BACKOFF_MS = 60_000;

/**
 * Calls `fn` and retries up to `MAX_ATTEMPTS` times on retryable OpenAI
 * errors (429 rate-limit / slow_down, 503 server_is_overloaded).
 *
 * Delay strategy:
 *  1. If the error carries a `Retry-After` response header, wait at least
 *     that many seconds (as required by the OpenAI API contract).
 *  2. Otherwise, use full-jitter exponential backoff:
 *       delay = random(0, min(MAX_BACKOFF_MS, baseMs * 2 ** attempt))
 */
async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  const baseMs = 500;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const isLast = attempt === MAX_ATTEMPTS - 1;

      // Only retry on known transient API errors.
      if (
        !APICallError.isInstance(err) ||
        !RETRYABLE_STATUS_CODES.has(err.statusCode ?? 0)
      ) {
        throw err;
      }

      if (isLast) {
        throw err;
      }

      // Honour Retry-After header when the API provides one.
      const retryAfterHeader =
        err.responseHeaders?.['retry-after'] ??
        err.responseHeaders?.['Retry-After'];

      let delayMs: number;
      if (retryAfterHeader) {
        const retryAfterSec = Number(retryAfterHeader);
        delayMs = Number.isFinite(retryAfterSec)
          ? retryAfterSec * 1_000
          : MAX_BACKOFF_MS;
      } else {
        // Full-jitter exponential backoff.
        const cap = Math.min(MAX_BACKOFF_MS, baseMs * 2 ** attempt);
        delayMs = Math.random() * cap;
      }

      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  // TypeScript control-flow: the loop always returns or throws.
  /* istanbul ignore next */
  throw new Error('withRetry: exhausted attempts without returning');
}

export interface ReplyDraft {
  model: string;
  body: string;
}

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
