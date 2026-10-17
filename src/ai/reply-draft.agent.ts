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

/** Retryable status codes per OpenAI best-practice guidance. */
const RETRYABLE_STATUS_CODES = new Set([429, 503]);

/** Maximum number of attempts (1 initial + 3 retries). */
const MAX_ATTEMPTS = 4;

/**
 * Calls `generateText` with automatic retry on 429 / 503 responses.
 * Reads the `Retry-After` header when present; otherwise uses exponential
 * backoff with full-jitter (base 1 s, capped at 32 s).
 */
async function generateTextWithRetry(
  params: Parameters<typeof generateText>[0],
): Promise<Awaited<ReturnType<typeof generateText>>> {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      return await generateText(params);
    } catch (err: unknown) {
      const isLast = attempt === MAX_ATTEMPTS - 1;

      // Extract a numeric status code from whatever the SDK surfaces.
      const status: number | undefined =
        err != null && typeof err === 'object' && 'status' in err
          ? (err as { status: unknown }).status as number
          : undefined;

      if (!RETRYABLE_STATUS_CODES.has(status ?? 0) || isLast) {
        throw err;
      }

      // Honour Retry-After (seconds) when the SDK exposes response headers.
      let delayMs: number | undefined;
      const headers: unknown =
        err != null && typeof err === 'object' && 'responseHeaders' in err
          ? (err as { responseHeaders: unknown }).responseHeaders
          : undefined;
      if (headers != null && typeof headers === 'object') {
        const retryAfter =
          (headers as Record<string, string | undefined>)['retry-after'] ??
          (headers as Record<string, string | undefined>)['Retry-After'];
        if (retryAfter !== undefined) {
          const seconds = parseFloat(retryAfter);
          if (Number.isFinite(seconds) && seconds >= 0) {
            delayMs = seconds * 1_000;
          }
        }
      }

      // Fall back to exponential backoff with full jitter: [0, base * 2^attempt).
      if (delayMs === undefined) {
        const baseCap = Math.min(1_000 * Math.pow(2, attempt), 32_000);
        delayMs = Math.random() * baseCap;
      }

      await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
    }
  }

  // Unreachable — TypeScript needs the explicit throw.
  throw new Error('generateTextWithRetry: exceeded max attempts');
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  // Note: `temperature` and `topP` are not supported by GPT-6 Astra and must
  // not be passed.  See OpenAI migration guidance for the GPT-6 family.
  const { text } = await generateTextWithRetry({
    model: openai(REPLY_DRAFT_MODEL),
    system: REPLY_DRAFT_SYSTEM_PROMPT,
    prompt: buildReplyDraftPrompt(input),
    maxOutputTokens: 500,
  });

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
