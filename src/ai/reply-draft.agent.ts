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
 * Retry wrapper that honours the Retry-After header on 429 / 503 responses.
 * Falls back to exponential back-off when the header is absent.
 *
 * @param fn        Async operation to attempt.
 * @param maxTries  Total number of attempts (default 4).
 */
async function withRetry<T>(fn: () => Promise<T>, maxTries = 4): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err: unknown) {
      attempt += 1;
      if (attempt >= maxTries) {
        throw err;
      }

      // Extract the HTTP status and Retry-After header from whatever the SDK
      // surfaces.  Both the native fetch Response and common SDK error shapes
      // are handled; anything unrecognised is re-thrown immediately.
      const status: number | undefined =
        (err as { status?: number })?.status ??
        (err as { statusCode?: number })?.statusCode;

      if (status !== 429 && status !== 503) {
        throw err;
      }

      const retryAfterRaw: string | undefined =
        (err as { headers?: Record<string, string> })?.headers?.[
          'retry-after'
        ] ??
        (err as { response?: { headers?: { get?: (h: string) => string | null } } })
          ?.response?.headers?.get?.('retry-after') ??
        undefined;

      let delayMs: number;
      if (retryAfterRaw !== undefined && retryAfterRaw !== null) {
        const seconds = Number(retryAfterRaw);
        delayMs = Number.isFinite(seconds)
          ? seconds * 1_000
          : // Retry-After may also be an HTTP-date — treat as 5 s fallback.
            5_000;
      } else {
        // Exponential back-off: 1 s, 2 s, 4 s, …
        delayMs = Math.pow(2, attempt - 1) * 1_000;
      }

      await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
    }
  }
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await withRetry(() =>
    generateText({
      model: openai(REPLY_DRAFT_MODEL),
      system: REPLY_DRAFT_SYSTEM_PROMPT,
      prompt: buildReplyDraftPrompt(input),
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
