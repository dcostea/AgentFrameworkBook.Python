"""Two independent runs: without a session the agent forgets the first response."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from shared.agent_examples import JSON_INSTRUCTIONS, STATELESS_FOLLOWUP_PROMPT, TREE_PROMPT
from shared.chat_clients import create_completion_client, create_sdk_client


async def run_example(agent: Agent):
    response = await agent.run(TREE_PROMPT)
    print(response.text)
    followup = await agent.run(STATELESS_FOLLOWUP_PROMPT)
    print(followup.text)
    return response, followup


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        agent = Agent(
            client=create_completion_client(sdk, provider),
            instructions=JSON_INSTRUCTIONS,
            name="RobotCarAgent",
            description="An AI agent that controls a robot car.",
        )
        await run_example(agent)


if __name__ == "__main__":
    asyncio.run(main())
