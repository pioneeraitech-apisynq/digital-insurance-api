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
 * Retry wrapper for generateText that handles 429 (rate-limit) and 503
 * (server overloaded) responses from the OpenAI API.
 *
 * Strategy (per OpenAI best-practice guidance):
 *  1. Read the `Retry-After` header from the error response and wait at least
 *     that many seconds before retrying.
 *  2. If the header is absent, fall back to exponential backoff with jitter
 *     (initial delay 1 s, doubling each attempt, capped at 32 s).
 */
const RETRYABLE_STATUS_CODES = new Set([429, 503]);
const MAX_ATTEMPTS = 5;

async function generateTextWithRetry(
  ...args: Parameters<typeof generateText>
): Promise<Awaited<ReturnType<typeof generateText>>> {
  let attempt = 0;

  while (true) {
    try {
      return await generateText(...args);
    } catch (err: unknown) {
      attempt += 1;

      // Extract HTTP status and Retry-After from the error if present.
      const status: number | undefined =
        err != null && typeof err === 'object' && 'status' in err
          ? (err as { status?: number }).status
          : undefined;

      const isRetryable = status !== undefined && RETRYABLE_STATUS_CODES.has(status);

      if (!isRetryable || attempt >= MAX_ATTEMPTS) {
        throw err;
      }

      // Honour the Retry-After header when present; otherwise use exponential
      // backoff with full jitter: delay ∈ [0, min(cap, base * 2^attempt)].
      const retryAfterSeconds: number | undefined =
        err != null && typeof err === 'object' && 'headers' in err
          ? parseRetryAfter((err as { headers?: Record<string, string> }).headers)
          : undefined;

      const backoffMs =
        retryAfterSeconds !== undefined
          ? retryAfterSeconds * 1000
          : Math.random() * Math.min(32_000, 1_000 * 2 ** attempt);

      await new Promise((resolve) => setTimeout(resolve, backoffMs));
    }
  }
}

function parseRetryAfter(
  headers: Record<string, string> | undefined,
): number | undefined {
  if (!headers) return undefined;
  const value =
    headers['retry-after'] ?? headers['Retry-After'] ?? headers['RETRY-AFTER'];
  if (value === undefined) return undefined;
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : undefined;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await generateTextWithRetry({
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
}
