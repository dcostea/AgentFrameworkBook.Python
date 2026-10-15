"""Send Robby's prompt to a locally running Ollama model."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Message
from agent_framework.ollama import OllamaChatClient
from ollama import AsyncClient

from shared.chat_clients import TREE_COMMAND

SYSTEM = """You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into basic moves you know.
Respond only with the permitted moves, without any additional explanations."""


async def main() -> None:
    sdk = AsyncClient(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    try:
        client = OllamaChatClient(client=sdk, model=os.environ.get("OLLAMA_MODEL", "gemma4:e4b"))
        messages = [Message("system", [SYSTEM]), Message("user", [f'Complex command:\n"{TREE_COMMAND}"'])]
        response = await client.get_response(messages)
        messages.extend(response.messages)
        print(response.text)
    finally:
        # Ollama does not yet expose a public async close method for its HTTP client.
        await sdk._client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
