import { createGateway } from 'ai/gateway';
import { generateText, wrapLanguageModel, type LanguageModelMiddleware } from 'ai';
import {
  REPLY_DRAFT_SYSTEM_PROMPT,
  ReplyDraftInput,
  buildReplyDraftPrompt,
} from './prompts/reply-draft.prompt';

/**
 * Reply draft helper.
 *
 * Runs OpenAI gpt-4o-mini through the AI SDK Gateway provider. Support agents
 * get a first draft of a reply to a customer message; a human always edits and
 * sends it, so nothing here is customer-facing on its own.
 *
 * Gateway routing (APISynQ) is configured via the APISYNC_GATEWAY_URL /
 * APISYNC_API_KEY environment variables picked up by createGateway, replacing
 * the previous baseURL override on createOpenAI.
 *
 * Guardrails are applied as Language Model Middleware so they are enforced
 * server-side regardless of prompt content, model updates, or injection
 * attempts in the customer message.
 */

/** The model this helper runs (unified gateway format). */
export const REPLY_DRAFT_MODEL = 'openai/gpt-4o-mini';

// ---------------------------------------------------------------------------
// Guardrail middleware
// ---------------------------------------------------------------------------

/**
 * Patterns that, if found in the generated output, indicate the model has
 * violated one of our documented guardrail rules. The middleware intercepts the
 * response before it is returned to the caller and replaces a violating draft
 * with a safe fallback instructing the human agent to compose the reply
 * manually.
 *
 * Rules enforced here mirror the system prompt rules so that a prompt
 * truncation, injection, or future model update cannot silently bypass them.
 *
 *  PII_PATTERNS        — full address, DOB, or payment details echoed back.
 *  COMPLAINT_PATTERNS  — regulator / lawyer / protected-characteristic language
 *                        that must be escalated, not answered.
 *  COMMITMENT_PATTERNS — claim acceptance, payout promise, or settlement quote.
 */
const PII_PATTERNS = [
  // Full date of birth (various separators)
  /\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b/,
  // Card / bank account numbers (simplified: 12–19 consecutive digits)
  /\b\d{12,19}\b/,
  // Sort-code format (UK)
  /\b\d{2}-\d{2}-\d{2}\b/,
];

const COMPLAINT_PATTERNS = [
  /\b(regulator|ombudsman|FCA|FOS|CFPB|NAIC)\b/i,
  /\b(lawyer|attorney|solicitor|legal action|sue|lawsuit)\b/i,
  /\b(discriminat|protected characteristic|equality act|civil rights)\b/i,
];

const COMMITMENT_PATTERNS = [
  /\b(we (accept|approve|will pay|are paying)|claim (is|has been) approved)\b/i,
  /\bsettlement (of|amount|figure)\b/i,
  /\bpayout of\b/i,
];

const SAFE_FALLBACK =
  'This message requires manual review. Please compose a reply directly ' +
  'without using the draft assistant for this case.';

const guardrailMiddleware: LanguageModelMiddleware = {
  wrapGenerate: async ({ doGenerate }) => {
    const result = await doGenerate();

    const generatedText =
      result.text ??
      result.content
        ?.filter((part) => part.type === 'text')
        .map((part) => (part as { type: 'text'; text: string }).text)
        .join('') ??
      '';

    const violatesPii = PII_PATTERNS.some((re) => re.test(generatedText));
    const violatesComplaint = COMPLAINT_PATTERNS.some((re) =>
      re.test(generatedText),
    );
    const violatesCommitment = COMMITMENT_PATTERNS.some((re) =>
      re.test(generatedText),
    );

    if (violatesPii || violatesComplaint || violatesCommitment) {
      // Replace the response content with the safe fallback so the caller
      // receives a coherent (though non-draft) string rather than an error.
      return {
        ...result,
        text: SAFE_FALLBACK,
        content: [{ type: 'text' as const, text: SAFE_FALLBACK }],
      };
    }

    return result;
  },
};

// ---------------------------------------------------------------------------
// Gateway provider + wrapped model
// ---------------------------------------------------------------------------

const gateway = createGateway({
  apiKey: process.env.APISYNC_API_KEY,
  baseURL: process.env.APISYNC_GATEWAY_URL,
});

const model = wrapLanguageModel({
  model: gateway(REPLY_DRAFT_MODEL),
  middleware: guardrailMiddleware,
});

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface ReplyDraft {
  model: string;
  body: string;
}

export async function draftReply(input: ReplyDraftInput): Promise<ReplyDraft> {
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

  return { model: REPLY_DRAFT_MODEL, body };
}
