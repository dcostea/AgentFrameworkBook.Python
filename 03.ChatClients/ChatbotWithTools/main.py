"""Automatic simulated motor-tool invocation with bounded model context."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import TOOLS_SYSTEM, create_completion_client, create_sdk_client
from shared.chatbot import motor_functions, run_chatbot


async def main() -> None:
    async with create_sdk_client("azure") as sdk:
        await run_chatbot(
            create_completion_client(sdk, "azure"), system=TOOLS_SYSTEM, reduce_history=True,
            options={"tool_choice": "auto", "allow_multiple_tool_calls": True, "tools": motor_functions()},
        )


if __name__ == "__main__":
    asyncio.run(main())
