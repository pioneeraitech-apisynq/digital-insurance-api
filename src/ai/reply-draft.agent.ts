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

/** HTTP status codes that are safe to retry (rate-limited or overloaded). */
const RETRYABLE_STATUS_CODES = new Set([429, 503]);

/** Maximum number of attempts before giving up. */
const MAX_ATTEMPTS = 3;

/** Base delay (ms) for exponential backoff when no Retry-After header is present. */
const BASE_BACKOFF_MS = 1_000;

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

/**
 * Returns the number of milliseconds to wait before the next attempt.
 *
 * Prefers the `Retry-After` header value (in seconds) supplied by OpenAI on
 * 429 / 503 responses. Falls back to exponential backoff when the header is
 * absent or unparseable.
 */
function retryDelayMs(attempt: number, retryAfterHeader?: string): number {
  if (retryAfterHeader) {
    const seconds = parseFloat(retryAfterHeader);
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.ceil(seconds * 1_000);
    }
  }
  // Exponential backoff: 1 s, 2 s, 4 s, …
  return BASE_BACKOFF_MS * Math.pow(2, attempt - 1);
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
    } catch (err: unknown) {
      lastError = err;

      // Only retry on rate-limit (429) or overload (503) responses.
      const status: number | undefined =
        err !== null &&
        typeof err === 'object' &&
        'status' in err &&
        typeof (err as Record<string, unknown>).status === 'number'
          ? (err as Record<string, unknown>).status as number
          : undefined;

      if (status === undefined || !RETRYABLE_STATUS_CODES.has(status)) {
        // Non-retryable error — surface immediately.
        throw err;
      }

      if (attempt === MAX_ATTEMPTS) {
        // All attempts exhausted.
        break;
      }

      // Read the Retry-After header that OpenAI attaches to 429 / 503 replies.
      const headers: Record<string, string> | undefined =
        err !== null &&
        typeof err === 'object' &&
        'responseHeaders' in err &&
        typeof (err as Record<string, unknown>).responseHeaders === 'object' &&
        (err as Record<string, unknown>).responseHeaders !== null
          ? (err as Record<string, unknown>).responseHeaders as Record<string, string>
          : undefined;

      const retryAfterHeader: string | undefined =
        headers?.['retry-after'] ?? headers?.['Retry-After'];

      const delay = retryDelayMs(attempt, retryAfterHeader);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}
