"""Persist a Python AgentSession as UTF-8 JSON, restore it, and continue."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent, AgentSession, InMemoryHistoryProvider
from shared.agent_examples import FOLLOWUP_PROMPT, ROBOT_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_completion_client, create_sdk_client

SESSION_PATH = Path(__file__).resolve().parent / "agent_session.json"


def save_session(session: AgentSession, path: Path = SESSION_PATH) -> None:
    path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(path: Path = SESSION_PATH) -> AgentSession:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict) or data.get("type") != "session"
        or not isinstance(data.get("session_id"), str) or not data["session_id"]
        or not isinstance(data.get("state"), dict)
    ):
        raise ValueError("Expected a Python AgentSession snapshot, not a C# session JSON file")
    return AgentSession.from_dict(data)


async def run_example(agent: Agent, *, path: Path = SESSION_PATH):
    session = agent.create_session()
    print(f"USER: {TREE_PROMPT}")
    response = await agent.run(TREE_PROMPT, session=session)
    print(f"ASSISTANT: {response.text}")
    print("\nSerializing session...")
    save_session(session, path)
    print("\nDeserializing session...")
    restored = load_session(path)
    print(f"USER: {FOLLOWUP_PROMPT}")
    followup = await agent.run(FOLLOWUP_PROMPT, session=restored)
    print(f"ASSISTANT: {followup.text}")
    return response, followup, restored


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        agent = Agent(
            client=create_completion_client(sdk, provider), instructions=ROBOT_INSTRUCTIONS,
            name="MotorsAgent", context_providers=[InMemoryHistoryProvider()],
        )
        await run_example(agent, path=SESSION_PATH)


if __name__ == "__main__":
    asyncio.run(main())
