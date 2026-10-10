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
 *
 * TODO (finding 4): When tool use is added to this agent, replace the
 * `generateText` / Chat Completions call with a Responses API call via
 * `openai` npm package's `client.responses.create` interface, as required
 * by GPT-6 Astra's tool-calling pathway. This requires adding `openai` as a
 * package dependency first.
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
 * Extracts a numeric seconds value from a Retry-After header string, which
 * may be either a delay-seconds integer or an HTTP-date.
 */
function retryAfterMs(header: string | undefined): number | null {
  if (!header) return null;
  const seconds = Number(header);
  if (!Number.isNaN(seconds) && seconds >= 0) return seconds * 1000;
  // HTTP-date form
  const at = Date.parse(header);
  if (!Number.isNaN(at)) return Math.max(0, at - Date.now());
  return null;
}

/**
 * Returns true for transient error codes that should be retried:
 *   - 429 slow_down   – traffic is ramping up too quickly
 *   - 503 server_is_overloaded – temporary model overload
 */
function isRetryable(err: unknown): boolean {
  if (err && typeof err === 'object') {
    const e = err as Record<string, unknown>;
    const code = e['code'] ?? (e['data'] as Record<string, unknown>)?.['code'];
    const status = e['status'] ?? e['statusCode'];
    if (code === 'slow_down' || code === 'server_is_overloaded') return true;
    // Treat bare 429 / 503 without a recognised code as retryable too.
    if (status === 429 || status === 503) return true;
  }
  return false;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const MAX_ATTEMPTS = 4;
  const BASE_BACKOFF_MS = 1_000;

  let lastError: unknown;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    if (attempt > 0) {
      // Respect Retry-After if present; otherwise use exponential backoff.
      const header =
        lastError &&
        typeof lastError === 'object' &&
        typeof (lastError as Record<string, unknown>)['retryAfter'] === 'string'
          ? ((lastError as Record<string, unknown>)['retryAfter'] as string)
          : undefined;
      const wait = retryAfterMs(header) ?? BASE_BACKOFF_MS * 2 ** (attempt - 1);
      await new Promise((resolve) => setTimeout(resolve, wait));
    }

    try {
      // Note: `temperature` is intentionally omitted — GPT-6 Astra does not
      // support custom temperature values. Consistency is enforced via the
      // system prompt instead.
      const { text } = await generateText({
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
    } catch (err) {
      lastError = err;
      if (!isRetryable(err) || attempt === MAX_ATTEMPTS - 1) {
        throw err;
      }
    }
  }

  // Unreachable — the loop always throws or returns — but satisfies TS.
  throw lastError;
}
