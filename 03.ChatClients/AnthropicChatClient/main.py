"""Send Robby's obstacle-avoidance prompt through the Anthropic provider."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Message
from agent_framework.anthropic import AnthropicClient
from anthropic import AsyncAnthropic

from shared.chat_clients import ROBOT_PROMPT, require_env


async def main() -> None:
    print(f"USER: {ROBOT_PROMPT}")
    async with AsyncAnthropic(api_key=require_env("ANTHROPIC_API_KEY")) as sdk:
        client = AnthropicClient(model=require_env("ANTHROPIC_MODEL_ID"), anthropic_client=sdk)
        response = await client.get_response(
            [Message("user", [ROBOT_PROMPT])], options={"max_tokens": 512},
        )
        print(f"\nAssistant (Agent Framework): {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
