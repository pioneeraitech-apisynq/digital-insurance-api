import { createOpenAI } from '@ai-sdk/openai';
import { generateText, APICallError } from 'ai';
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

/** HTTP status codes that OpenAI returns for transient, retryable conditions. */
const RETRYABLE_STATUS_CODES = new Set([
  429, // rate-limited / slow_down
  503, // server_is_overloaded
]);

/** Maximum number of attempts before giving up. */
const MAX_ATTEMPTS = 3;

/** Base delay (ms) for exponential backoff when no Retry-After header is present. */
const BASE_BACKOFF_MS = 1_000;

/** Returns the number of milliseconds the caller should wait before the next attempt. */
function retryDelayMs(attempt: number, responseHeaders?: Record<string, string>): number {
  const retryAfterHeader =
    responseHeaders?.['retry-after'] ?? responseHeaders?.['Retry-After'];
  if (retryAfterHeader) {
    const seconds = parseFloat(retryAfterHeader);
    if (Number.isFinite(seconds) && seconds > 0) {
      return seconds * 1_000;
    }
  }
  // Exponential backoff: 1 s, 2 s, 4 s, …
  return BASE_BACKOFF_MS * Math.pow(2, attempt);
}

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  let lastError: unknown;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
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

      // Only retry on known transient OpenAI errors.
      const isRetryable =
        APICallError.isInstance(err) &&
        typeof err.statusCode === 'number' &&
        RETRYABLE_STATUS_CODES.has(err.statusCode);

      if (!isRetryable || attempt === MAX_ATTEMPTS - 1) {
        throw err;
      }

      const headers = APICallError.isInstance(err)
        ? (err.responseHeaders as Record<string, string> | undefined)
        : undefined;

      const delay = retryDelayMs(attempt, headers);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  // Unreachable, but satisfies the TypeScript control-flow checker.
  throw lastError;
}
