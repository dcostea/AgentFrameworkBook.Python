"""Require operator approval before simulated backward and stop motor moves."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_sdk_client
from shared.tool_examples import build_agent, create_motor_tools, run_with_approval

PROMPT = "Complex command: Danger ahead! Stop! Full back!"


async def main() -> None:
    async with create_sdk_client() as sdk:
        agent = build_agent(sdk, tools=create_motor_tools(require_approval=True))
        response = await run_with_approval(agent, PROMPT)
        print(f"\nRESPONSE: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
