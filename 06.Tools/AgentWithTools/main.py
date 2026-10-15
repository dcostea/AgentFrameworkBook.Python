"""Automatically execute simulated motor tools for an obstacle-avoidance command."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_sdk_client
from shared.color_helper import Color, print_colored_line
from shared.tool_examples import INSTRUCTIONS, OBSTACLE_PROMPT, build_agent


async def main() -> None:
    async with create_sdk_client() as sdk:
        agent = build_agent(sdk, instructions=INSTRUCTIONS + "\nCall the matching tools.")
        print_colored_line(f"USER: {OBSTACLE_PROMPT}", Color.YELLOW)
        response = await agent.run(OBSTACLE_PROMPT)
        print_colored_line(f"ASSISTANT: {response.text}", Color.GREEN)


if __name__ == "__main__":
    asyncio.run(main())
