"""Add information-level logging middleware to an OpenAI chat client."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Message

from shared.chat_clients import ROBOT_PROMPT, create_completion_client, create_sdk_client
from shared.chat_middleware import LoggingChatMiddleware


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"USER: {ROBOT_PROMPT}")
    async with create_sdk_client() as sdk:
        client = create_completion_client(sdk, middleware=[LoggingChatMiddleware()])
        response = await client.get_response([Message("user", [ROBOT_PROMPT])])
        print(f"\nAssistant: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
