"""Durable atomic JSON history in this sample's runtime directory.

Reuse the printed --session-id on a later invocation to resume a cold session.
No source C# generated history dump is used to initialize writable history.
"""

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent, AgentSession
from shared.chat_clients import create_completion_client, create_sdk_client
from shared.memory_providers import (
    CustomFileBasedChatHistoryProvider, ROBOT_INSTRUCTIONS, ROBOT_PROMPTS,
    print_history, run_prompts,
)

RUNTIME = Path(__file__).resolve().parent / "runtime"


def build_agent(client, storage_path=RUNTIME):
    history = CustomFileBasedChatHistoryProvider(storage_path)
    return Agent(
        client=client, name="RobotCarAgent", instructions=ROBOT_INSTRUCTIONS,
        description="An agent that assists a robot with the basic moves.",
        context_providers=[history],
    )


async def main(provider="openai", session_id=None):
    async with create_sdk_client(provider) as sdk:
        agent = build_agent(create_completion_client(sdk, provider))
        session = AgentSession(session_id=session_id)
        print(f"Resume with --session-id {session.session_id}")
        history = agent.context_providers[0]
        prompts = (ROBOT_PROMPTS[1],) if await history.get_messages(session.session_id) else ROBOT_PROMPTS
        await run_prompts(agent, session, prompts)
        print_history(await history.get_messages(session.session_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    parser.add_argument("--session-id", help="ID printed by an earlier run")
    args = parser.parse_args()
    asyncio.run(main(args.provider, args.session_id))
