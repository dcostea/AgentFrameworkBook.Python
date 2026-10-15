"""Supply conversation history explicitly, without creating an agent session."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent, Message
from shared.agent_examples import EXAMPLE_MOVES, FOLLOWUP_PROMPT, JSON_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_completion_client, create_sdk_client


def build_conversation() -> list[Message]:
    return [
        Message("user", [TREE_PROMPT]),
        Message("assistant", [EXAMPLE_MOVES]),
        Message("user", [FOLLOWUP_PROMPT]),
    ]


async def run_example(agent: Agent):
    conversation = build_conversation()
    response = await agent.run(conversation)
    print(response.text)
    conversation.extend(response.messages)
    return response, conversation


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        agent = Agent(client=create_completion_client(sdk, provider), instructions=JSON_INSTRUCTIONS)
        await run_example(agent)


if __name__ == "__main__":
    asyncio.run(main())
