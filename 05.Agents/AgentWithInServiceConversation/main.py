"""Create, use, inspect, and always delete a provider-managed conversation."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from shared.agent_examples import FOLLOWUP_PROMPT, ROBOT_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_responses_client, create_sdk_client


async def print_history(sdk, conversation_id: str) -> None:
    async for item in sdk.conversations.items.list(conversation_id, order="asc"):
        if item.type != "message":
            continue
        print(f"  {item.role} [{item.id}]:")
        for content in item.content:
            if text := getattr(content, "text", None):
                print(text)
        print()


async def run_example(sdk, agent: Agent):
    # The Responses adapter omits instructions when a conv_* ID is supplied.
    # Seed the empty service conversation so the first turn still has its persona.
    conversation = await sdk.conversations.create(items=[{
        "type": "message", "role": "system", "content": ROBOT_INSTRUCTIONS,
    }])
    conversation_id = conversation.id
    try:
        session = agent.get_session(service_session_id=conversation_id)
        print(f"USER: {TREE_PROMPT}")
        response = await agent.run(TREE_PROMPT, session=session)
        print(f"ASSISTANT: {response.text}")
        print(f"USER: {FOLLOWUP_PROMPT}")
        followup = await agent.run(FOLLOWUP_PROMPT, session=session)
        print(f"ASSISTANT: {followup.text}")
        print("HISTORY:")
        await print_history(sdk, conversation_id)
        return response, followup, session
    finally:
        deleted = await sdk.conversations.delete(conversation_id)
        print(f"  Deleted: {deleted.deleted}")


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        agent = Agent(
            client=create_responses_client(sdk, provider), instructions=ROBOT_INSTRUCTIONS,
            default_options={"store": True},
        )
        await run_example(sdk, agent)


if __name__ == "__main__":
    asyncio.run(main())
