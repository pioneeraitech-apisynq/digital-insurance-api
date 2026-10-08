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
 * Retry configuration for transient OpenAI errors (429 rate-limit / 503
 * overload). When the response includes a Retry-After header we honour it;
 * otherwise we use truncated exponential backoff with full jitter.
 */
const RETRY_MAX_ATTEMPTS = 4;
const RETRY_BASE_DELAY_MS = 500;
const RETRY_MAX_DELAY_MS = 16_000;

/** Returns true for status codes that are safe to retry. */
function isRetryableStatus(status: number): boolean {
  return status === 429 || status === 503;
}

/**
 * Extracts the wait time (in ms) from a Retry-After header value, which may
 * be a delta-seconds integer or an HTTP-date string. Returns null when the
 * header is absent or unparseable.
 */
function retryAfterMs(retryAfterHeader: string | undefined): number | null {
  if (!retryAfterHeader) return null;

  const seconds = Number(retryAfterHeader);
  if (!Number.isNaN(seconds) && seconds >= 0) {
    return seconds * 1_000;
  }

  const date = new Date(retryAfterHeader).getTime();
  if (!Number.isNaN(date)) {
    const delta = date - Date.now();
    return delta > 0 ? delta : 0;
  }

  return null;
}

/** Sleeps for the given number of milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Runs `fn`, retrying on 429 / 503 errors. Respects the Retry-After header
 * when present; falls back to exponential backoff with full jitter otherwise.
 */
async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  let attempt = 0;

  while (true) {
    try {
      return await fn();
    } catch (err: unknown) {
      attempt += 1;

      // Only retry on recognisable rate-limit / overload errors.
      const status =
        err != null && typeof err === 'object' && 'status' in err
          ? (err as { status: unknown }).status
          : undefined;

      const retryable =
        typeof status === 'number' && isRetryableStatus(status);

      if (!retryable || attempt >= RETRY_MAX_ATTEMPTS) {
        throw err;
      }

      // Prefer the server-supplied back-off, fall back to exponential + jitter.
      const serverHint =
        err != null && typeof err === 'object' && 'headers' in err
          ? retryAfterMs(
              (err as { headers?: Record<string, string> }).headers?.[
                'retry-after'
              ],
            )
          : null;

      const backoff = Math.min(
        RETRY_BASE_DELAY_MS * 2 ** (attempt - 1),
        RETRY_MAX_DELAY_MS,
      );
      const jitter = Math.random() * backoff;
      const waitMs = serverHint !== null ? serverHint : jitter;

      await sleep(waitMs);
    }
  }
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await withRetry(() =>
    generateText({
      model: openai(REPLY_DRAFT_MODEL),
      system: REPLY_DRAFT_SYSTEM_PROMPT,
      prompt: buildReplyDraftPrompt(input),
      // Note: temperature is intentionally omitted — gpt-6-astra does not
      // support custom temperature values.
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
