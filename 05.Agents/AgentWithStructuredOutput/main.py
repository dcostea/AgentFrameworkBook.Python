"""Generate the original Steps/Move/Value JSON schema as a typed response."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from pydantic import BaseModel, ConfigDict
from shared.agent_examples import ROBOT_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_completion_client, create_sdk_client


class StepItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Move: str
    Value: str | None


class StepsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Steps: list[StepItem]


def create_agent(client) -> Agent:
    return Agent(
        client=client, instructions=ROBOT_INSTRUCTIONS, name="RobotCarAgent",
        description="An agent that assists a robot with the basic moves.",
        default_options={"temperature": 0.4, "max_tokens": 300, "response_format": StepsResponse},
    )


async def run_example(agent: Agent):
    response = await agent.run(TREE_PROMPT)
    result = response.value
    if not isinstance(result, StepsResponse):
        raise ValueError("The agent did not return a valid StepsResponse")
    print(result.model_dump_json())
    return response


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        await run_example(create_agent(create_completion_client(sdk, provider)))


if __name__ == "__main__":
    asyncio.run(main())
