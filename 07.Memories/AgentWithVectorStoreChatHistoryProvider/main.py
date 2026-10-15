"""Metadata collection history: earliest five messages, chronological order.

Despite the source name, this sample does NOT perform embedding/similarity
search. Its collection holds full serialized messages only in process memory.
A resumed session can reload history only while that same store still exists.
"""

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from shared.chat_clients import create_completion_client, create_sdk_client
from shared.memory_providers import (
    InMemoryChatHistoryStore, VectorStoreChatHistoryProvider,
    ROBOT_INSTRUCTIONS, ROBOT_PROMPTS, print_history, run_prompts,
)


def build_agent(client, store=None):
    history = VectorStoreChatHistoryProvider(
        store if store is not None else InMemoryChatHistoryStore(), top_results=5,
    )
    return Agent(
        client=client, name="RobotCarAgent", instructions=ROBOT_INSTRUCTIONS,
        description="An agent that assists a robot with the basic moves.",
        context_providers=[history],
    )


async def main(provider="openai"):
    async with create_sdk_client(provider) as sdk:
        agent = build_agent(create_completion_client(sdk, provider))
        session = agent.create_session()
        await run_prompts(agent, session, ROBOT_PROMPTS)
        print_history(await agent.context_providers[0].all_messages(session.session_id))
        print("Storage is in memory only; a new process starts with empty history.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    asyncio.run(main(parser.parse_args().provider))
