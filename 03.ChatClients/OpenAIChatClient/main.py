"""Compare the OpenAI SDK and Agent Framework Chat Completions clients."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Message

from shared.chat_clients import ROBOT_PROMPT, create_completion_client, create_sdk_client, model_for


async def main() -> None:
    print(f"USER: {ROBOT_PROMPT}")
    async with create_sdk_client() as sdk:
        response = await sdk.chat.completions.create(
            model=model_for(), messages=[{"role": "user", "content": ROBOT_PROMPT}],
        )
        print(f"\nAssistant (OpenAI ChatClient): {response.choices[0].message.content}")
        client = create_completion_client(sdk)
        response = await client.get_response([Message("user", [ROBOT_PROMPT])])
        print(f"\nAssistant (Agent Framework): {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
