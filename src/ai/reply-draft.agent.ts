import { gateway } from '@ai-sdk/gateway';
import { generateText } from 'ai';
import {
  REPLY_DRAFT_SYSTEM_PROMPT,
  ReplyDraftInput,
  buildReplyDraftPrompt,
} from './prompts/reply-draft.prompt';

/**
 * Reply draft helper.
 *
 * Runs gpt-4o-mini through the Vercel AI Gateway provider. Support agents get
 * a first draft of a reply to a customer message; a human always edits and
 * sends it, so nothing here is customer-facing on its own.
 *
 * The AI Gateway provider handles routing, data-class policy enforcement, and
 * call recording centrally, removing the need for a bespoke OPENAI_BASE_URL.
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'openai/gpt-4o-mini';

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
  const { text } = await generateText({
    model: gateway(REPLY_DRAFT_MODEL),
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
