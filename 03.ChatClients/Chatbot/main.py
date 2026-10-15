"""An interactive Azure chatbot with explicit conversation history."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_completion_client, create_sdk_client
from shared.chatbot import run_chatbot


async def main() -> None:
    async with create_sdk_client("azure") as sdk:
        await run_chatbot(create_completion_client(sdk, "azure"))


if __name__ == "__main__":
    asyncio.run(main())
