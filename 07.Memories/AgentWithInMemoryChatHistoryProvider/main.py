"""Session history reduced before retrieval to the last three non-tool messages."""

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from shared.chat_clients import create_completion_client, create_sdk_client
from shared.memory_providers import (
    MessageCountingHistoryProvider, ROBOT_INSTRUCTIONS, ROBOT_PROMPTS,
    print_history, run_prompts,
)


def build_agent(client):
    return Agent(
        client=client, name="RobotCarAgent", instructions=ROBOT_INSTRUCTIONS,
        description="An agent that assists a robot with the basic moves.",
        context_providers=[MessageCountingHistoryProvider(target_count=3)],
    )


async def main(provider="openai"):
    async with create_sdk_client(provider) as sdk:
        agent = build_agent(create_completion_client(sdk, provider))
        session = agent.create_session()
        await run_prompts(agent, session, ROBOT_PROMPTS)
        print_history(session.state["reduced-history"]["messages"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    asyncio.run(main(parser.parse_args().provider))
