import { Injectable, NotFoundException } from '@nestjs/common';
import {
  ClaimResource,
  listClaimsByPolicyNumber,
} from '../clients/claims-api.client';
import { draftReply, ReplyDraft } from '../ai/reply-draft.agent';
import { DraftReplyDto } from './dto/draft-reply.dto';

export interface Customer {
  id: string;
  name: string;
  email: string;
  policyNumbers: string[];
}

@Injectable()
export class CustomersService {
  private readonly customers = new Map<string, Customer>([
    [
      'cust_001',
      {
        id: 'cust_001',
        name: 'Ada Lovelace',
        email: 'ada@example.com',
        policyNumbers: ['POL-1001', 'POL-1042'],
      },
    ],
    [
      'cust_002',
      {
        id: 'cust_002',
        name: 'Alan Turing',
        email: 'alan@example.com',
        policyNumbers: ['POL-2087'],
      },
    ],
  ]);

  findOne(customerId: string): Customer {
    const customer = this.customers.get(customerId);
    if (!customer) {
      throw new NotFoundException(`Customer ${customerId} not found`);
    }
    return customer;
  }

  /**
   * Every claim this customer has filed, gathered from the internal Claims
   * API. The Claims API is indexed by policy number, so one lookup runs per
   * policy the customer holds and the results are merged.
   */
  async findClaims(customerId: string): Promise<ClaimResource[]> {
    const customer = this.findOne(customerId);

    const perPolicy = await Promise.all(
      customer.policyNumbers.map((policyNumber) =>
        listClaimsByPolicyNumber(policyNumber),
      ),
    );

    return perPolicy.flat();
  }

  /**
   * Draft a reply to a message the customer sent. The draft goes to a support
   * agent, who edits and sends it; nothing here reaches the customer directly.
   *
   * @param abortSignal - Optional signal that cancels the in-flight LLM call
   *   when the HTTP client disconnects, preventing unnecessary token spend.
   */
  async draftMessageReply(
    customerId: string,
    dto: DraftReplyDto,
    abortSignal?: AbortSignal,
  ): Promise<ReplyDraft> {
    const customer = this.findOne(customerId);
    const claims = await this.findClaims(customerId);

    return draftReply(
      {
        customerName: customer.name,
        message: dto.message,
        policyNumbers: customer.policyNumbers,
        openClaimCount: claims.filter((claim) => claim.status !== 'closed')
          .length,
        topic: dto.topic,
      },
      abortSignal,
    );
  }
}
