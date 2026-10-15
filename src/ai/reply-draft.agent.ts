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

/** Transient HTTP status codes that warrant a retry. */
const RETRYABLE_STATUSES = new Set([429, 503]);

/** Maximum number of attempts (1 initial + 3 retries). */
const MAX_ATTEMPTS = 4;

/**
 * Extracts the number of milliseconds to wait from an error thrown by the AI
 * SDK.  Reads the `Retry-After` response header when present (value is in
 * seconds per RFC 9110); falls back to exponential backoff otherwise.
 */
function retryDelayMs(error: unknown, attempt: number): number {
  const BACKOFF_BASE_MS = 500;

  if (error instanceof Error) {
    // The Vercel AI SDK attaches `responseHeaders` to API errors.
    const headers: Record<string, string> | undefined =
      (error as unknown as { responseHeaders?: Record<string, string> })
        .responseHeaders;

    if (headers) {
      const retryAfter =
        headers['retry-after'] ?? headers['Retry-After'] ?? headers['RETRY-AFTER'];
      if (retryAfter) {
        const seconds = Number(retryAfter);
        if (Number.isFinite(seconds) && seconds >= 0) {
          return seconds * 1000;
        }
      }
    }
  }

  // Exponential backoff with full jitter: random in [0, base * 2^attempt).
  const cap = BACKOFF_BASE_MS * Math.pow(2, attempt);
  return Math.random() * cap;
}

/** Returns true if the error represents a retryable HTTP status. */
function isRetryable(error: unknown): boolean {
  if (error instanceof Error) {
    const status: unknown =
      (error as unknown as { status?: unknown }).status;
    return typeof status === 'number' && RETRYABLE_STATUSES.has(status);
  }
  return false;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  let lastError: unknown;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      const { text } = await generateText({
        model: openai(REPLY_DRAFT_MODEL),
        system: REPLY_DRAFT_SYSTEM_PROMPT,
        prompt: buildReplyDraftPrompt(input),
        // temperature is not supported by gpt-6-astra; omitted intentionally.
        maxOutputTokens: 500,
      });

      const body = text.trim();
      if (!body) {
        throw new Error('Reply draft came back empty');
      }

      return { model: REPLY_DRAFT_MODEL, body };
    } catch (err) {
      if (isRetryable(err) && attempt < MAX_ATTEMPTS - 1) {
        lastError = err;
        const delay = retryDelayMs(err, attempt);
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }
      throw err;
    }
  }

  // Unreachable in practice, but satisfies the TypeScript control-flow checker.
  throw lastError;
}
