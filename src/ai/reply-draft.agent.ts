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
 * Runs OpenAI gpt-6-astra through the Vercel AI SDK. Support agents get a
 * first draft of a reply to a customer message; a human always edits and sends
 * it, so nothing here is customer-facing on its own.
 *
 * The base URL is read from OPENAI_BASE_URL rather than left at the SDK
 * default, so the same build can be pointed at the APISynQ gateway (which
 * records the call and applies the data-class policy) instead of talking to
 * api.openai.com directly.
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-6-astra';

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

/**
 * Retryable wrapper around an async operation.
 *
 * On a 429 (rate-limit / slow_down) or 503 (server_is_overloaded) response the
 * OpenAI SDK surfaces an error whose `status` field carries the HTTP status and
 * whose `headers` map may include a `retry-after` value (seconds). We honour
 * that header when present; otherwise we wait with capped exponential back-off.
 */
const RETRYABLE_STATUSES = new Set([429, 503]);
const MAX_ATTEMPTS = 4;
const BASE_DELAY_MS = 1_000;
const MAX_DELAY_MS = 30_000;

async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err: unknown) {
      attempt += 1;

      const status: number | undefined =
        err != null && typeof err === 'object' && 'status' in err
          ? (err as { status: unknown }).status as number
          : undefined;

      const isRetryable = status !== undefined && RETRYABLE_STATUSES.has(status);

      if (!isRetryable || attempt >= MAX_ATTEMPTS) {
        throw err;
      }

      // Respect Retry-After when the API provides it.
      const retryAfterHeader: string | undefined =
        err != null &&
        typeof err === 'object' &&
        'headers' in err &&
        err.headers != null &&
        typeof err.headers === 'object' &&
        'retry-after' in (err.headers as object)
          ? String((err.headers as Record<string, unknown>)['retry-after'])
          : undefined;

      const retryAfterMs = retryAfterHeader
        ? parseFloat(retryAfterHeader) * 1_000
        : Math.min(BASE_DELAY_MS * 2 ** (attempt - 1), MAX_DELAY_MS);

      await new Promise<void>((resolve) => setTimeout(resolve, retryAfterMs));
    }
  }
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await withRetry(() =>
    generateText({
      model: openai(REPLY_DRAFT_MODEL),
      system: REPLY_DRAFT_SYSTEM_PROMPT,
      prompt: buildReplyDraftPrompt(input),
      // temperature is not set: gpt-6-astra does not support custom
      // temperature, top_p, or logprobs values.
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
