"""Compare native Azure Responses with the Agent Framework Responses client."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Message

from shared.chat_clients import ROBOT_PROMPT, create_responses_client, create_sdk_client, model_for


async def main() -> None:
    print(f"USER: {ROBOT_PROMPT}")
    async with create_sdk_client("azure") as sdk:
        response = await sdk.responses.create(model=model_for("azure", responses=True), input=ROBOT_PROMPT)
        print(f"\nAssistant (Azure Responses): {response.output_text}")
        client = create_responses_client(sdk, "azure")
        response = await client.get_response([Message("user", [ROBOT_PROMPT])])
        print(f"\nAssistant (Agent Framework): {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
