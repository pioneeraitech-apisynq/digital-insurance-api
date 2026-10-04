import { Body, Controller, Get, Param, Post, Req } from '@nestjs/common';
import { Request } from 'express';
import { CustomersService } from './customers.service';
import { DraftReplyDto } from './dto/draft-reply.dto';

@Controller('v1/customers')
export class CustomersController {
  constructor(private readonly customersService: CustomersService) {}

  @Get(':customerId')
  findOne(@Param('customerId') customerId: string) {
    return this.customersService.findOne(customerId);
  }

  @Get(':customerId/claims')
  findClaims(@Param('customerId') customerId: string) {
    return this.customersService.findClaims(customerId);
  }

  @Post(':customerId/messages/draft-reply')
  draftReply(
    @Param('customerId') customerId: string,
    @Body() dto: DraftReplyDto,
    @Req() req: Request,
  ) {
    // Build an AbortSignal that fires when the HTTP client disconnects so the
    // in-flight LLM call is cancelled and tokens are not wasted.
    const controller = new AbortController();
    req.socket.once('close', () => controller.abort());

    return this.customersService.draftMessageReply(
      customerId,
      dto,
      controller.signal,
    );
  }
}
