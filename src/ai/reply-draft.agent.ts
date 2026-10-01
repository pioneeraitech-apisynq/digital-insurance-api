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
 *
 * Security: OPENAI_BASE_URL is validated against an explicit allow-list of
 * trusted origins at module load time. This prevents an attacker who can
 * influence the runtime environment from redirecting requests to an arbitrary
 * host (DNS rebinding / SSRF). Add legitimate gateway origins to
 * ALLOWED_BASE_URL_ORIGINS below when provisioning new environments.
 */

/** Trusted origins that OPENAI_BASE_URL may point at. */
const ALLOWED_BASE_URL_ORIGINS = new Set([
  'https://api.openai.com',
  'https://gateway.apisync.io',
]);

function validateBaseURL(raw: string | undefined): string | undefined {
  if (raw === undefined) {
    return undefined;
  }
  let origin: string;
  try {
    origin = new URL(raw).origin;
  } catch {
    throw new Error(
      `OPENAI_BASE_URL is not a valid URL: "${raw}". ` +
        'Set it to a trusted gateway origin.',
    );
  }
  if (!ALLOWED_BASE_URL_ORIGINS.has(origin)) {
    throw new Error(
      `OPENAI_BASE_URL origin "${origin}" is not in the allow-list. ` +
        'Add it to ALLOWED_BASE_URL_ORIGINS in reply-draft.agent.ts only ' +
        'after confirming it is a trusted APISynQ gateway endpoint.',
    );
  }
  return raw;
}

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-4o-mini';

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: validateBaseURL(process.env.OPENAI_BASE_URL),
});

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
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
}
