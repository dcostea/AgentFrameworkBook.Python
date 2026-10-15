"""Let an auditor inspect the motors agent's moves through shared local history."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent, InMemoryHistoryProvider
from shared.agent_examples import AUDIT_PROMPT, AUDITOR_INSTRUCTIONS, ROBOT_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_completion_client, create_sdk_client


def create_agents(client) -> tuple[Agent, Agent]:
    # Both agents read/write the same session-state namespace, not separate histories.
    motors = Agent(
        client=client, instructions=ROBOT_INSTRUCTIONS, name="MotorsAgent",
        context_providers=[InMemoryHistoryProvider("shared_history")],
    )
    auditor = Agent(
        client=client, instructions=AUDITOR_INSTRUCTIONS, name="AuditorAgent",
        context_providers=[InMemoryHistoryProvider("shared_history")],
    )
    return motors, auditor


async def run_example(motors: Agent, auditor: Agent):
    session = motors.create_session()
    print(f"USER (MOTORS): {TREE_PROMPT}")
    response = await motors.run(TREE_PROMPT, session=session)
    print(f"ASSISTANT (MOTORS): {response.text}")
    print(f"USER (AUDITOR): {AUDIT_PROMPT}")
    audit = await auditor.run(AUDIT_PROMPT, session=session)
    print(f"ASSISTANT (AUDITOR): {audit.text}")
    return response, audit, session


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        motors, auditor = create_agents(create_completion_client(sdk, provider))
        await run_example(motors, auditor)


if __name__ == "__main__":
    asyncio.run(main())
