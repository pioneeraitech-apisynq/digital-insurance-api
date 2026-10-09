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

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

/** Statuses that are transient and warrant a retry. */
const RETRYABLE_STATUSES = new Set([429, 503]);

/**
 * Calls `fn` and retries on 429 / 503 responses.
 *
 * The delay is taken from the `Retry-After` response header when present;
 * otherwise exponential backoff (1 s, 2 s, 4 s, …) is used.
 * After `maxAttempts` the last error is re-thrown.
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts = 4,
): Promise<T> {
  let attempt = 0;

  while (true) {
    attempt += 1;
    try {
      return await fn();
    } catch (err: unknown) {
      const isLastAttempt = attempt >= maxAttempts;

      // Extract HTTP status and Retry-After from whatever the SDK throws.
      const status: number | undefined =
        (err as Record<string, unknown>)?.status as number | undefined ??
        (err as Record<string, unknown>)?.statusCode as number | undefined;

      const retryAfterHeader: string | undefined =
        (err as Record<string, unknown>)?.responseHeaders?.['retry-after'] as
          | string
          | undefined ??
        (err as Record<string, unknown>)?.headers?.['retry-after'] as
          | string
          | undefined;

      if (isLastAttempt || !RETRYABLE_STATUSES.has(status as number)) {
        throw err;
      }

      // Honour Retry-After (seconds) when provided; fall back to exponential backoff.
      const retryAfterSec = retryAfterHeader ? parseFloat(retryAfterHeader) : NaN;
      const delayMs = Number.isFinite(retryAfterSec) && retryAfterSec >= 0
        ? retryAfterSec * 1000
        : Math.pow(2, attempt - 1) * 1000; // 1 s, 2 s, 4 s, …

      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await withRetry(() =>
    generateText({
      model: openai(REPLY_DRAFT_MODEL),
      system: REPLY_DRAFT_SYSTEM_PROMPT,
      prompt: buildReplyDraftPrompt(input),
      // temperature is intentionally omitted: GPT-6 Astra (and future models)
      // do not support custom temperature or top_p values, so omitting it here
      // keeps the call forward-compatible with any model upgrade.
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
