"""Start a Responses API background operation and poll its continuation token."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent
from shared.agent_examples import BACKGROUND_INSTRUCTIONS, TREE_PROMPT
from shared.chat_clients import create_responses_client, create_sdk_client


def provider_response(response):
    # AgentResponse wraps a ChatResponse, which wraps the SDK Response.
    chat_response = response.raw_representation
    return getattr(chat_response, "raw_representation", chat_response)


def print_response(label: str, response) -> None:
    raw = provider_response(response)
    print(f"\n[{label}] Assistant:")
    print(f"Response Status: {getattr(raw, 'status', None)}")
    print(f"Has Continuation Token: {response.continuation_token is not None}")
    print(response.text)


async def run_example(agent: Agent, *, max_attempts: int = 60, delay: float = 1.0):
    if max_attempts < 1 or delay < 0:
        raise ValueError("max_attempts must be positive and delay nonnegative")
    session = agent.create_session()
    print(f"User:\n{TREE_PROMPT}")
    response = await agent.run(TREE_PROMPT, session=session, options={"background": True, "store": True})
    print_response("INITIAL", response)
    for _ in range(max_attempts):
        if response.continuation_token is None:
            break
        await asyncio.sleep(delay)
        print(".", end="", flush=True)
        response = await agent.run(
            [], session=session, options={"continuation_token": response.continuation_token},
        )
    if response.continuation_token is not None:
        response_id = response.continuation_token.get("response_id", "unknown")
        raise TimeoutError(
            f"Polling timeout after {max_attempts} attempts. Response {response_id} is still running; "
            "retrieve or cancel it with the provider's Responses API."
        )
    print_response("FINAL", response)
    raw = provider_response(response)
    if getattr(raw, "status", None) in {"failed", "cancelled", "incomplete"}:
        detail = getattr(raw, "error", None) or getattr(raw, "incomplete_details", None)
        raise RuntimeError(f"Background response ended with status {raw.status}: {detail}")
    return response, session


async def main(provider: str = "openai") -> None:
    async with create_sdk_client(provider) as sdk:
        agent = Agent(client=create_responses_client(sdk, provider), instructions=BACKGROUND_INSTRUCTIONS)
        await run_example(agent)


if __name__ == "__main__":
    asyncio.run(main())
