"""Request one tool call per model roundtrip, with an opt-in parallel mode."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_sdk_client
from shared.tool_examples import OBSTACLE_PROMPT, build_agent, print_tools


async def main(parallel: bool = False) -> None:
    async with create_sdk_client() as sdk:
        agent = build_agent(sdk, options={"allow_multiple_tool_calls": parallel})
        response = await agent.run(OBSTACLE_PROMPT)
        print_tools(response.messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", action="store_true", help="Allow several tool calls in a model response")
    asyncio.run(main(parser.parse_args().parallel))
