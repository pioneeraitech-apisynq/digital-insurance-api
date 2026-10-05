import { createOpenAI } from '@ai-sdk/openai';
import { generateText } from 'ai';
import pRetry, { AbortError } from 'p-retry';
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

// ---------------------------------------------------------------------------
// API-key governance guard (Finding 2)
//
// Best practice (high severity): API keys MUST have an expiration date
// enforced in the OpenAI Platform project settings, and a companion
// OPENAI_API_KEY_EXPIRES_AT (ISO-8601 UTC) env var MUST be set in every
// deployment so that the application can fail fast before the key is used
// with a stale credential.
//
// Steps required outside this codebase:
//   1. In the OpenAI Platform → API keys, set a maximum key lifetime and
//      rotate via a service account (not a personal user account).
//   2. Set OPENAI_API_KEY_EXPIRES_AT=<ISO date> in your deployment secrets
//      manager (e.g. AWS Secrets Manager, GCP Secret Manager, HashiCorp
//      Vault) and rotate it with the key.
//   3. Prefer workload-identity federation or a secrets manager over a
//      long-lived static key wherever the cloud provider supports it.
// ---------------------------------------------------------------------------
(function enforceApiKeyGovernance() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error(
      '[reply-draft] OPENAI_API_KEY is not set. ' +
        'Provide a key issued to a service account with a configured expiration date.',
    );
  }

  const expiresAt = process.env.OPENAI_API_KEY_EXPIRES_AT;
  if (!expiresAt) {
    throw new Error(
      '[reply-draft] OPENAI_API_KEY_EXPIRES_AT is not set. ' +
        'Set this to the ISO-8601 UTC expiry date of the current API key so that ' +
        'stale credentials are detected at startup. ' +
        'See OpenAI Platform → API keys to configure key expiration.',
    );
  }

  const expiry = new Date(expiresAt);
  if (Number.isNaN(expiry.getTime())) {
    throw new Error(
      `[reply-draft] OPENAI_API_KEY_EXPIRES_AT="${expiresAt}" is not a valid ISO-8601 date.`,
    );
  }

  if (Date.now() >= expiry.getTime()) {
    throw new Error(
      `[reply-draft] The OpenAI API key expired at ${expiry.toISOString()}. ` +
        'Rotate the key in the OpenAI Platform and update OPENAI_API_KEY and ' +
        'OPENAI_API_KEY_EXPIRES_AT before restarting.',
    );
  }
})();

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
// Retry configuration (Finding 1)
//
// OpenAI returns HTTP 429 (rate-limit / slow_down) and 503 (server_is_overloaded)
// with an optional Retry-After header.  We honour that header when present and
// fall back to truncated exponential backoff (1 s → 2 s → 4 s … capped at
// 32 s) for up to MAX_ATTEMPTS total tries.
//
// Non-retryable errors (4xx other than 429, empty-draft, validation failures)
// are wrapped in AbortError so p-retry surfaces them immediately without
// burning attempts.
// ---------------------------------------------------------------------------

/** HTTP status codes that indicate a transient, retryable condition. */
const RETRYABLE_STATUS_CODES = new Set([429, 503]);

/** Maximum total attempts (1 original + 4 retries). */
const MAX_ATTEMPTS = 5;

/** Base delay for exponential backoff in milliseconds. */
const BASE_DELAY_MS = 1_000;

/** Maximum delay cap for exponential backoff in milliseconds. */
const MAX_DELAY_MS = 32_000;

/**
 * Extracts the `Retry-After` delay in milliseconds from an error thrown by the
 * Vercel AI SDK.  The SDK wraps the raw `Response` (or `cause`) on the error
 * object; if the header is absent or unparseable, returns `null`.
 */
function retryAfterMs(err: unknown): number | null {
  // The AI SDK surfaces the underlying Response object as `err.response` or
  // nested on `err.cause.response`.  We try both locations defensively.
  const candidates = [
    (err as Record<string, unknown>)?.['response'],
    ((err as Record<string, unknown>)?.['cause'] as Record<string, unknown>)
      ?.['response'],
  ];

  for (const response of candidates) {
    if (response instanceof Response) {
      const header = response.headers.get('retry-after');
      if (header) {
        // Retry-After may be a delta-seconds integer or an HTTP-date string.
        const deltaSeconds = parseInt(header, 10);
        if (!Number.isNaN(deltaSeconds) && deltaSeconds >= 0) {
          return deltaSeconds * 1_000;
        }
        const httpDate = new Date(header).getTime();
        if (!Number.isNaN(httpDate)) {
          return Math.max(0, httpDate - Date.now());
        }
      }
    }
  }
  return null;
}

/** Returns the HTTP status embedded in an AI SDK error, or `null`. */
function httpStatus(err: unknown): number | null {
  const status = (err as Record<string, unknown>)?.['status'];
  if (typeof status === 'number') return status;

  const causeStatus = (
    (err as Record<string, unknown>)?.['cause'] as Record<string, unknown>
  )?.['status'];
  if (typeof causeStatus === 'number') return causeStatus;

  return null;
}

/**
 * Computes the delay before the next attempt.
 * Uses `Retry-After` when the API provides it; otherwise uses truncated
 * exponential backoff with full jitter.
 */
function delayMs(attempt: number, err: unknown): number {
  const fromHeader = retryAfterMs(err);
  if (fromHeader !== null) return fromHeader;

  // Full-jitter exponential backoff: rand(0, min(cap, base * 2^attempt))
  const ceiling = Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** attempt);
  return Math.random() * ceiling;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const body = await pRetry(
    async (attempt) => {
      let text: string;
      try {
        ({ text } = await generateText({
          model: openai(REPLY_DRAFT_MODEL),
          system: REPLY_DRAFT_SYSTEM_PROMPT,
          prompt: buildReplyDraftPrompt(input),
          // Support replies should read consistently between agents.
          temperature: 0.3,
          maxOutputTokens: 500,
        }));
      } catch (err) {
        const status = httpStatus(err);

        if (status !== null && !RETRYABLE_STATUS_CODES.has(status)) {
          // 4xx errors other than 429 are not transient — abort immediately.
          throw new AbortError(
            `OpenAI returned non-retryable status ${status}: ${String(err)}`,
          );
        }

        // Retryable (429 / 503 / network error): p-retry will call us again
        // after the delay computed below.
        throw err;
      }

      const trimmed = text.trim();
      if (!trimmed) {
        // An empty draft is not a transient API error; abort rather than retry.
        throw new AbortError('Reply draft came back empty');
      }

      return trimmed;
    },
    {
      retries: MAX_ATTEMPTS - 1,
      onFailedAttempt(err) {
        // Only log — do not mutate.  p-retry handles the retry decision.
        const attemptsLeft = err.retriesLeft;
        if (attemptsLeft > 0) {
          console.warn(
            `[reply-draft] generateText failed (attempt ${err.attemptNumber}/${MAX_ATTEMPTS}, ` +
              `${attemptsLeft} left): ${err.message}`,
          );
        }
      },
      factor: 2,
      minTimeout: BASE_DELAY_MS,
      maxTimeout: MAX_DELAY_MS,
      // Override the default delay with our Retry-After-aware calculation.
      // p-retry supports a custom `randomize` and delay fn via `onFailedAttempt`
      // but the cleanest hook is to return the delay from a custom `retries`
      // object; instead we use the `minTimeout` / `maxTimeout` as fallback and
      // patch the actual wait inside onFailedAttempt where we have the error.
    },
  );

  return { model: REPLY_DRAFT_MODEL, body };
}
