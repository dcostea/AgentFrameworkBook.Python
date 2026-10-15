"""Intercept motor calls, execute them explicitly, and return correlated results."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_sdk_client
from shared.tool_examples import build_agent, create_motor_tools, run_manual_tools

PROMPT = 'Complex command:\n"Go left and right then stop."'


async def main() -> None:
    async with create_sdk_client() as sdk:
        tools = create_motor_tools()
        agent = build_agent(sdk, tools=tools, manual=True,
                            options={"allow_multiple_tool_calls": True, "tool_choice": "auto"})
        response = await run_manual_tools(agent, PROMPT, tools)
        print(f"\nAssistant: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
