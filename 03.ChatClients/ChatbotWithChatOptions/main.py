"""Chat options and schema-constrained output (requires a compatible model)."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_completion_client, create_sdk_client
from shared.chatbot import StepsResponse, run_chatbot


def chat_options() -> dict:
    return {
        "temperature": 0.4,
        "max_tokens": int(os.getenv("CHATBOT_MAX_OUTPUT_TOKENS", "512")),
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "stop": ["END"],
        "response_format": StepsResponse,
    }


async def main() -> None:
    async with create_sdk_client("azure") as sdk:
        await run_chatbot(create_completion_client(sdk, "azure"), options=chat_options())


if __name__ == "__main__":
    asyncio.run(main())
