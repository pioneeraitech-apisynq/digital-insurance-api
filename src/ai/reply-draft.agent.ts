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
 * Calls `generateText` with exponential backoff for transient 429 / 503
 * errors.  The `Retry-After` header is honoured when present; otherwise the
 * delay doubles each attempt starting at 1 s (cap: 32 s, max 6 attempts).
 */
async function generateTextWithRetry(
  params: Parameters<typeof generateText>[0],
): Promise<Awaited<ReturnType<typeof generateText>>> {
  const MAX_ATTEMPTS = 6;
  const BASE_DELAY_MS = 1_000;
  const MAX_DELAY_MS = 32_000;

  let attempt = 0;
  while (true) {
    try {
      return await generateText(params);
    } catch (err: unknown) {
      attempt += 1;

      // Surface any error that is not a rate-limit / overload signal, or that
      // has exhausted all retries, immediately.
      const status: number | undefined =
        err != null &&
        typeof err === 'object' &&
        'status' in err &&
        typeof (err as { status: unknown }).status === 'number'
          ? (err as { status: number }).status
          : undefined;

      const isRetryable = status === 429 || status === 503;
      if (!isRetryable || attempt >= MAX_ATTEMPTS) {
        throw err;
      }

      // Respect Retry-After when the SDK surfaces it on the error object.
      let delayMs: number | undefined;
      if (
        err != null &&
        typeof err === 'object' &&
        'headers' in err &&
        err.headers != null &&
        typeof err.headers === 'object'
      ) {
        const retryAfter = (err.headers as Record<string, string | undefined>)[
          'retry-after'
        ];
        if (retryAfter !== undefined) {
          const seconds = Number(retryAfter);
          if (!Number.isNaN(seconds)) {
            delayMs = seconds * 1_000;
          }
        }
      }

      if (delayMs === undefined) {
        // Exponential backoff: 1 s, 2 s, 4 s, 8 s, 16 s, 32 s …
        delayMs = Math.min(BASE_DELAY_MS * 2 ** (attempt - 1), MAX_DELAY_MS);
      }

      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await generateTextWithRetry({
    model: openai(REPLY_DRAFT_MODEL),
    system: REPLY_DRAFT_SYSTEM_PROMPT,
    prompt: buildReplyDraftPrompt(input),
    // temperature is not supported by GPT-6 Astra and has been removed.
    maxOutputTokens: 500,
  });

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
