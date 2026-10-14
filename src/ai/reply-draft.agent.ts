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

// ---------------------------------------------------------------------------
// Retry helper
// ---------------------------------------------------------------------------

/** Status codes that are safe and recommended to retry. */
const RETRYABLE_STATUS = new Set([429, 503]);

/** Maximum number of attempts (1 initial + N retries). */
const MAX_ATTEMPTS = 4;

/** Base delay used for exponential backoff when no Retry-After header is present. */
const BASE_BACKOFF_MS = 500;

/**
 * Parses the `Retry-After` value from an error's response headers if the
 * underlying HTTP layer exposes them, returning the wait duration in ms.
 * Falls back to `undefined` when the header is absent or unparseable.
 */
function retryAfterMs(err: unknown): number | undefined {
  if (err === null || typeof err !== 'object') return undefined;

  // The Vercel AI SDK (and many HTTP clients) surface response headers on the
  // error object under a `response` or `headers` property.
  const candidate =
    (err as Record<string, unknown>)['response'] ??
    (err as Record<string, unknown>)['headers'];

  if (candidate === null || typeof candidate !== 'object') return undefined;

  const headers = candidate as Record<string, unknown>;

  // `headers` may be a Headers instance or a plain object.
  const raw: unknown =
    typeof (headers as { get?: unknown }).get === 'function'
      ? (headers as { get(k: string): unknown }).get('retry-after')
      : headers['retry-after'] ?? headers['Retry-After'];

  if (raw === null || raw === undefined) return undefined;

  const seconds = Number(raw);
  return Number.isFinite(seconds) ? seconds * 1_000 : undefined;
}

/** Returns `true` when the error represents a 429 or 503 HTTP response. */
function isRetryable(err: unknown): boolean {
  if (err === null || typeof err !== 'object') return false;
  const status = (err as Record<string, unknown>)['status'];
  return typeof status === 'number' && RETRYABLE_STATUS.has(status);
}

/**
 * Calls `fn` and retries up to `MAX_ATTEMPTS - 1` times on 429 / 503 errors,
 * waiting at least as long as the `Retry-After` header dictates, or using
 * exponential backoff when the header is missing.
 */
async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (!isRetryable(err) || attempt === MAX_ATTEMPTS - 1) {
        throw err;
      }

      lastError = err;

      const waitMs =
        retryAfterMs(err) ??
        BASE_BACKOFF_MS * Math.pow(2, attempt); // 500 ms, 1 s, 2 s, …

      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }

  // Unreachable in practice, but satisfies the type-checker.
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
      // NOTE: temperature / top_p are intentionally omitted — GPT-6 Astra
      // does not support custom values for these parameters.
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
