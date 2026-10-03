import { createGateway } from '@ai-sdk/gateway';
import { generateText } from 'ai';
import {
  REPLY_DRAFT_SYSTEM_PROMPT,
  ReplyDraftInput,
  buildReplyDraftPrompt,
} from './prompts/reply-draft.prompt';

/**
 * Reply draft helper.
 *
 * Runs gpt-4o-mini through the Vercel AI Gateway, which records the call and
 * applies the data-class policy defined in the APISynQ gateway configuration.
 * Support agents get a first draft of a reply to a customer message; a human
 * always edits and sends it, so nothing here is customer-facing on its own.
 */

/** The model this helper runs. */
export const REPLY_DRAFT_MODEL = 'gpt-4o-mini';

const gateway = createGateway();

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(
  input: ReplyDraftInput,
  signal?: AbortSignal,
): Promise<ReplyDraft> {
  const { text } = await generateText({
    model: gateway(REPLY_DRAFT_MODEL),
    system: REPLY_DRAFT_SYSTEM_PROMPT,
    prompt: buildReplyDraftPrompt(input),
    // Support replies should read consistently between agents.
    temperature: 0.3,
    maxOutputTokens: 500,
    abortSignal: signal,
  });

  const body = text.trim();
  if (!body) {
    throw new Error('Reply draft came back empty');
  }

  return { model: REPLY_DRAFT_MODEL, body };
}
