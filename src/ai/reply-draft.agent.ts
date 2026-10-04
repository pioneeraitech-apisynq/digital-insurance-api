import { createGateway } from 'ai';
import { generateText } from 'ai';
import {
  REPLY_DRAFT_SYSTEM_PROMPT,
  ReplyDraftInput,
  buildReplyDraftPrompt,
} from './prompts/reply-draft.prompt';

/**
 * Reply draft helper.
 *
 * Runs gpt-4o-mini through the Vercel AI SDK's unified AI Gateway provider.
 * Support agents get a first draft of a reply to a customer message; a human
 * always edits and sends it, so nothing here is customer-facing on its own.
 *
 * The gateway base URL is read from OPENAI_BASE_URL (kept for compatibility)
 * so the same build can be pointed at the APISynQ gateway, which records the
 * call and applies the data-class policy, instead of talking to the upstream
 * provider directly.
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-4o-mini';

const gateway = createGateway({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(
  input: ReplyDraftInput,
  abortSignal?: AbortSignal,
): Promise<ReplyDraft> {
  const { text } = await generateText({
    model: gateway(REPLY_DRAFT_MODEL),
    system: REPLY_DRAFT_SYSTEM_PROMPT,
    prompt: buildReplyDraftPrompt(input),
    // Support replies should read consistently between agents.
    temperature: 0.3,
    maxOutputTokens: 500,
    abortSignal,
  });

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
