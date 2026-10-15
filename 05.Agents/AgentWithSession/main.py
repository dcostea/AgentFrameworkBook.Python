"""Reuse a local AgentSession so the second run remembers the first response."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from shared.agent_examples import FOLLOWUP_PROMPT, ROBOT_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_completion_client, create_sdk_client


async def run_example(agent: Agent):
    session = agent.create_session()
    print(f"USER: {TREE_PROMPT}")
    response = await agent.run(TREE_PROMPT, session=session)
    print(f"ASSISTANT: {response.text}")
    print(f"USER: {FOLLOWUP_PROMPT}")
    followup = await agent.run(FOLLOWUP_PROMPT, session=session)
    print(f"ASSISTANT: {followup.text}")
    return response, followup, session


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        agent = Agent(client=create_completion_client(sdk, provider), instructions=ROBOT_INSTRUCTIONS)
        await run_example(agent)


if __name__ == "__main__":
    asyncio.run(main())
