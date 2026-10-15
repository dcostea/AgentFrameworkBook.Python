"""Information-level request logging without exposing prompt or credential data."""

import logging
from time import perf_counter

from agent_framework import ChatContext, ChatMiddleware

logger = logging.getLogger("chat_clients")


class LoggingChatMiddleware(ChatMiddleware):
    async def process(self, context: ChatContext, call_next) -> None:
        started = perf_counter()
        logger.info("Chat request: %d messages, streaming=%s", len(context.messages), context.stream)
        try:
            await call_next()
        except Exception:
            logger.error("Chat request failed after %.3fs", perf_counter() - started)
            raise
        logger.info("Chat response received after %.3fs", perf_counter() - started)
