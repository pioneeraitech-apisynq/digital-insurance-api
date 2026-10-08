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
//
// OpenAI returns:
//   429  { code: 'slow_down' }        – traffic ramping too fast
//   503  { code: 'server_is_overloaded' } – temporary model overload
//
// Both may carry a Retry-After header (seconds). When present we wait at
// least that long; when absent we use capped exponential backoff.
// ---------------------------------------------------------------------------

const RETRYABLE_STATUSES = new Set([429, 503]);
const MAX_ATTEMPTS = 4;
const BASE_DELAY_MS = 500;
const MAX_DELAY_MS = 30_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  let attempt = 0;

  while (true) {
    try {
      return await fn();
    } catch (err: unknown) {
      attempt += 1;

      // Surface the error immediately when we've exhausted attempts or the
      // error is not a retryable HTTP status.
      const status: number | undefined =
        err != null && typeof err === 'object' && 'status' in err
          ? (err as { status: unknown }).status as number
          : undefined;

      if (attempt >= MAX_ATTEMPTS || !RETRYABLE_STATUSES.has(status as number)) {
        throw err;
      }

      // Respect Retry-After when the server provides it.
      let delayMs: number | undefined;
      if (err != null && typeof err === 'object' && 'responseHeaders' in err) {
        const headers = (err as { responseHeaders: unknown }).responseHeaders;
        if (headers != null && typeof headers === 'object' && 'retry-after' in headers) {
          const retryAfter = (headers as Record<string, string>)['retry-after'];
          const seconds = Number(retryAfter);
          if (Number.isFinite(seconds) && seconds > 0) {
            delayMs = seconds * 1_000;
          }
        }
      }

      // Fall back to capped exponential backoff when no header is present.
      if (delayMs === undefined) {
        delayMs = Math.min(BASE_DELAY_MS * 2 ** (attempt - 1), MAX_DELAY_MS);
      }

      await sleep(delayMs);
    }
  }
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await withRetry(() =>
    generateText({
      model: openai(REPLY_DRAFT_MODEL),
      system: REPLY_DRAFT_SYSTEM_PROMPT,
      prompt: buildReplyDraftPrompt(input),
      // temperature is not supported by gpt-6-astra and has been removed.
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
