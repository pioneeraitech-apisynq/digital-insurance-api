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
 * Runs OpenAI gpt-4o-mini through the Vercel AI SDK Gateway provider. Support
 * agents get a first draft of a reply to a customer message; a human always
 * edits and sends it, so nothing here is customer-facing on its own.
 *
 * The Gateway provider abstracts provider-specific details and routes requests
 * through the APISynQ gateway (which records the call and applies the
 * data-class policy) without requiring a manual baseURL override.
 */

/** The model this helper runs, in Gateway unified format. */
export const REPLY_DRAFT_MODEL = 'openai/gpt-4o-mini';

const gateway = createGateway({
  apiKey: process.env.OPENAI_API_KEY,
});

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
