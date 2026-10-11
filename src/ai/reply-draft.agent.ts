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
 * Retryable wrapper around generateText.
 *
 * On a 429 (rate-limited / slow_down) or 503 (server_is_overloaded) response
 * the function waits for the duration specified in the `Retry-After` header
 * (interpreted as seconds), falling back to exponential backoff when the
 * header is absent. Up to MAX_RETRIES attempts are made before the error is
 * re-thrown.
 */
const MAX_RETRIES = 4;
const BASE_BACKOFF_MS = 500;

function getRetryAfterMs(err: unknown): number | null {
  // The Vercel AI SDK surfaces HTTP details on the raw response object attached
  // to the thrown error. We also accept a plain object with a `status` field so
  // callers can pass through errors from other shapes.
  if (err && typeof err === 'object') {
    const e = err as Record<string, unknown>;

    const status =
      typeof e['status'] === 'number'
        ? e['status']
        : typeof e['statusCode'] === 'number'
          ? e['statusCode']
          : null;

    if (status !== 429 && status !== 503) {
      return null; // Not a retriable status — do not retry.
    }

    // Look for a Retry-After header value (seconds).
    const headers = e['responseHeaders'] ?? e['headers'];
    if (headers && typeof headers === 'object') {
      const h = headers as Record<string, unknown>;
      const retryAfter = h['retry-after'] ?? h['Retry-After'];
      if (retryAfter !== undefined) {
        const secs = Number(retryAfter);
        if (!isNaN(secs) && secs >= 0) {
          return secs * 1000;
        }
      }
    }

    // Retriable status but no header — signal that backoff should be used.
    return -1;
  }
  return null;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const { text } = await generateText({
        model: openai(REPLY_DRAFT_MODEL),
        system: REPLY_DRAFT_SYSTEM_PROMPT,
        prompt: buildReplyDraftPrompt(input),
        // gpt-6-astra does not accept a custom temperature — omitted.
        maxOutputTokens: 500,
      });

      const body = text.trim();
      if (!body) {
        throw new Error('Reply draft came back empty');
      }

      return { model: REPLY_DRAFT_MODEL, body };
    } catch (err) {
      lastError = err;

      const retryAfterMs = getRetryAfterMs(err);
      if (retryAfterMs === null || attempt === MAX_RETRIES) {
        // Non-retriable error, or we have exhausted all retries.
        throw err;
      }

      const waitMs =
        retryAfterMs >= 0
          ? retryAfterMs
          : BASE_BACKOFF_MS * 2 ** attempt; // Exponential backoff fallback.

      await sleep(waitMs);
    }
  }

  // Unreachable, but TypeScript needs a guaranteed return/throw.
  throw lastError;
}
