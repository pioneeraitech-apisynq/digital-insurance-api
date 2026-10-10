import { AISDKError, generateText, LanguageModel } from 'ai';
import {
  REPLY_DRAFT_SYSTEM_PROMPT,
  ReplyDraftInput,
  buildReplyDraftPrompt,
} from './prompts/reply-draft.prompt';

/**
 * Reply draft helper.
 *
 * Accepts a provider-agnostic AI SDK Core `LanguageModel` so the underlying
 * provider (OpenAI, the APISynQ gateway, or any other) can be configured and
 * injected by the caller rather than hard-wired here.  Support agents get a
 * first draft of a reply to a customer message; a human always edits and sends
 * it, so nothing here is customer-facing on its own.
 *
 * The base URL is read from OPENAI_BASE_URL rather than left at the SDK
 * default, so the same build can be pointed at the APISynQ gateway (which
 * records the call and applies the data-class policy) instead of talking to
 * api.openai.com directly.
 */

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(
  model: LanguageModel,
  input: ReplyDraftInput,
): Promise<ReplyDraft> {
  try {
    const { text } = await generateText({
      model,
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

    return { model: model.modelId, body };
  } catch (err) {
    if (AISDKError.isInstance(err)) {
      // Re-throw with a clear label so the NestJS layer can log / respond
      // to SDK-level failures (invalid response data, abort, timeouts, …)
      // distinctly from auth or network errors.
      throw new Error(`AI SDK error during reply draft [${err.name}]: ${err.message}`, {
        cause: err,
      });
    }
    throw err;
  }
}
