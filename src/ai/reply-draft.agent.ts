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
 * Runs OpenAI gpt-6-luna through the Vercel AI SDK. Support agents get a
 * first draft of a reply to a customer message; a human always edits and sends
 * it, so nothing here is customer-facing on its own.
 *
 * The base URL is read from OPENAI_BASE_URL rather than left at the SDK
 * default, so the same build can be pointed at the APISynQ gateway (which
 * records the call and applies the data-class policy) instead of talking to
 * api.openai.com directly.
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-6-luna';

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
// OpenAI may return HTTP 429 (rate-limit / slow_down) or 503 (server
// overloaded) with an optional Retry-After header.  Both statuses must be
// retried rather than surfaced immediately to the caller.
//
//   • If Retry-After is present its value (seconds) is used as the delay.
//   • Otherwise exponential backoff with full jitter is used, capped at 32 s.
//
// A maximum of MAX_ATTEMPTS total attempts is made (1 original + 4 retries).
// Any other error, or exhaustion of all attempts, is re-thrown.
// ---------------------------------------------------------------------------

const MAX_ATTEMPTS = 5;
const RETRYABLE_STATUSES = new Set([429, 503]);

function retryableStatusOf(err: unknown): number | null {
  // The Vercel AI SDK surfaces the HTTP status on the thrown error object.
  // We check the two most common shapes; extend if the SDK changes.
  if (err && typeof err === 'object') {
    const e = err as Record<string, unknown>;
    const status =
      typeof e['status'] === 'number'
        ? e['status']
        : typeof e['statusCode'] === 'number'
          ? e['statusCode']
          : null;
    if (status !== null && RETRYABLE_STATUSES.has(status)) {
      return status;
    }
  }
  return null;
}

function retryAfterMs(err: unknown, attempt: number): number {
  // Try to read Retry-After from the error (SDK may attach response headers).
  if (err && typeof err === 'object') {
    const e = err as Record<string, unknown>;
    const headers = e['responseHeaders'] ?? e['headers'];
    if (headers && typeof headers === 'object') {
      const h = headers as Record<string, unknown>;
      const raw = h['retry-after'] ?? h['Retry-After'];
      if (raw !== undefined) {
        const secs = Number(raw);
        if (Number.isFinite(secs) && secs > 0) {
          return secs * 1000;
        }
      }
    }
  }
  // Exponential backoff with full jitter: random value in [0, min(cap, base * 2^attempt)]
  const capMs = 32_000;
  const baseMs = 500;
  const ceiling = Math.min(capMs, baseMs * Math.pow(2, attempt));
  return Math.random() * ceiling;
}

async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (retryableStatusOf(err) === null) {
        // Non-retryable error — propagate immediately.
        throw err;
      }
      if (attempt < MAX_ATTEMPTS - 1) {
        const delayMs = retryAfterMs(err, attempt);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }
  throw lastErr;
}

// ---------------------------------------------------------------------------

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await withRetry(() =>
    generateText({
      model: openai(REPLY_DRAFT_MODEL),
      system: REPLY_DRAFT_SYSTEM_PROMPT,
      prompt: buildReplyDraftPrompt(input),
      // temperature is intentionally omitted: GPT-6 family models do not
      // support custom temperature / top_p values.
      maxOutputTokens: 500,
    }),
  );

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
