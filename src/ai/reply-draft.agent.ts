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
 * Runs OpenAI gpt-5.6-sol through the Vercel AI SDK. Support agents get a
 * first draft of a reply to a customer message; a human always edits and sends
 * it, so nothing here is customer-facing on its own.
 *
 * The base URL is read from OPENAI_BASE_URL rather than left at the SDK
 * default, so the same build can be pointed at the APISynQ gateway (which
 * records the call and applies the data-class policy) instead of talking to
 * api.openai.com directly.
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-5.6-sol';

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

/** Maximum number of attempts before giving up (1 initial + 3 retries). */
const MAX_ATTEMPTS = 4;
/** Base delay in ms for exponential backoff when no Retry-After header is present. */
const BASE_BACKOFF_MS = 500;

/**
 * Retryable HTTP status codes returned by the OpenAI API.
 * 429 – rate-limited / slow_down
 * 503 – server_is_overloaded
 */
const RETRYABLE_STATUS_CODES = new Set([429, 503]);

/** Pause execution for `ms` milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Extract a numeric delay (in ms) from a thrown OpenAI API error.
 *
 * The Vercel AI SDK surfaces HTTP errors as objects that carry a `status`
 * field and, when present, a `responseHeaders` map.  We read the standard
 * `retry-after` header (value in whole seconds) first; if it is absent we
 * fall back to exponential back-off based on the attempt number.
 */
function retryDelayMs(error: unknown, attempt: number): number | null {
  if (
    error === null ||
    typeof error !== 'object' ||
    !RETRYABLE_STATUS_CODES.has((error as { status?: number }).status ?? 0)
  ) {
    return null; // not retryable
  }

  const headers: Record<string, string> | undefined = (
    error as { responseHeaders?: Record<string, string> }
  ).responseHeaders;

  const retryAfterHeader =
    headers?.['retry-after'] ?? headers?.['Retry-After'];

  if (retryAfterHeader !== undefined) {
    const seconds = parseFloat(retryAfterHeader);
    if (!Number.isNaN(seconds)) {
      // Add a small jitter (up to 200 ms) to avoid thundering-herd on the gateway.
      return Math.ceil(seconds * 1000) + Math.floor(Math.random() * 200);
    }
  }

  // Exponential backoff: 500 ms, 1 s, 2 s, 4 s, …
  return BASE_BACKOFF_MS * Math.pow(2, attempt - 1);
}

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
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
    } catch (err) {
      lastError = err;

      const delay = retryDelayMs(err, attempt);
      if (delay === null || attempt === MAX_ATTEMPTS) {
        // Not a retryable error, or we've exhausted all attempts.
        throw err;
      }

      await sleep(delay);
    }
  }

  // TypeScript requires a return/throw after the loop even though the loop
  // always exits via return or throw internally.
  throw lastError;
}
