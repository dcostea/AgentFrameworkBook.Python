"""Stream text deltas and token usage from a stateless agent run."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from shared.agent_examples import JSON_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_completion_client, create_sdk_client


async def run_example(agent: Agent):
    stream = agent.run(TREE_PROMPT, stream=True)
    async for update in stream:
        print(update.text, end="", flush=True)
        for content in update.contents:
            if content.type == "usage" and content.usage_details:
                usage = content.usage_details
                print(f"\n\nInput Tokens: {usage.get('input_token_count')}")
                print(f"Output Tokens: {usage.get('output_token_count')}")
                print(f"Total Tokens: {usage.get('total_token_count')}")
    print()
    return await stream.get_final_response()


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        agent = Agent(client=create_completion_client(sdk, provider), instructions=JSON_INSTRUCTIONS)
        await run_example(agent)


if __name__ == "__main__":
    asyncio.run(main())
