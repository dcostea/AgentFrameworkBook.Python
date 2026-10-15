"""Stream JSON output and preserve each completed response in history."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_completion_client, create_sdk_client
from shared.chatbot import StepsResponse, run_chatbot


async def main() -> None:
    async with create_sdk_client("azure") as sdk:
        await run_chatbot(
            create_completion_client(sdk, "azure"), stream=True, options={"response_format": StepsResponse},
        )


if __name__ == "__main__":
    asyncio.run(main())
