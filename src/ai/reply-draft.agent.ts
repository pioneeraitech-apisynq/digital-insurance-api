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

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

// ---------------------------------------------------------------------------
// Retry / back-off helpers
// ---------------------------------------------------------------------------

/** Statuses that OpenAI asks callers to retry (rate-limit or overload). */
const RETRYABLE_STATUSES = new Set([429, 503]);

/** Maximum number of attempts (1 initial + 2 retries). */
const MAX_ATTEMPTS = 3;

/** Base delay for exponential back-off when no Retry-After header is present. */
const BASE_BACKOFF_MS = 1_000;

/** Upper bound on computed back-off to avoid very long waits. */
const MAX_BACKOFF_MS = 30_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Extract the number of milliseconds to wait from a `Retry-After` header.
 * The header may be a delta-seconds integer or an HTTP-date string.
 * Returns `null` when the header is absent or unparseable.
 */
function parseRetryAfterMs(headers: Record<string, string> | undefined): number | null {
  const raw = headers?.['retry-after'] ?? headers?.['Retry-After'];
  if (!raw) return null;

  const seconds = Number(raw);
  if (!Number.isNaN(seconds) && seconds >= 0) {
    return Math.min(seconds * 1_000, MAX_BACKOFF_MS);
  }

  const date = Date.parse(raw);
  if (!Number.isNaN(date)) {
    const delta = date - Date.now();
    return Math.min(Math.max(delta, 0), MAX_BACKOFF_MS);
  }

  return null;
}

/**
 * Calls `generateText` and retries on 429 / 503 responses.
 *
 * - Honous the `Retry-After` response header when present.
 * - Falls back to exponential back-off (1 s → 2 s → …, capped at 30 s)
 *   when the header is absent.
 * - Gives up and rethrows after MAX_ATTEMPTS total attempts.
 */
async function generateTextWithRetry(
  params: Parameters<typeof generateText>[0],
): Promise<Awaited<ReturnType<typeof generateText>>> {
  let attempt = 0;

  while (true) {
    attempt += 1;
    try {
      return await generateText(params);
    } catch (err: unknown) {
      // Surface non-retryable errors immediately.
      const status: number | undefined =
        (err as { status?: number })?.status ??
        (err as { statusCode?: number })?.statusCode;

      if (!RETRYABLE_STATUSES.has(status as number) || attempt >= MAX_ATTEMPTS) {
        throw err;
      }

      // Determine how long to wait before the next attempt.
      const headers: Record<string, string> | undefined =
        (err as { responseHeaders?: Record<string, string> })?.responseHeaders;

      const retryAfterMs = parseRetryAfterMs(headers);
      const backoffMs =
        retryAfterMs ?? Math.min(BASE_BACKOFF_MS * 2 ** (attempt - 1), MAX_BACKOFF_MS);

      await sleep(backoffMs);
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
