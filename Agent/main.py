"""Agent - Python mirror of the C# Agent/Program.cs.

The most basic sample: create an agent with instructions, then run two
queries against it.
"""

# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from pathlib import Path
from random import randint
from typing import Annotated

# Allow importing the sibling "shared" package (Agents/shared).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_framework import Message, tool
from pydantic import Field

from shared import create_chat_client

# NOTE: approval_mode="never_require" is for sample brevity.
@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

async def main() -> None:
    """Run a basic prompt using a selected built-in client."""
    client = create_chat_client()

    message = Message("user", contents=["What's the weather in Amsterdam and in Paris?"])
    stream = "false"
    print(f"User: {message.text}")

    if stream:
        response_stream = client.get_response([message], stream=True, options={"tools": get_weather})
        print("Assistant: ", end="")
        async for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="")
        print("")
    else:
        print(f"Assistant: {await client.get_response([message], stream=False, options={'tools': get_weather})}")


if __name__ == "__main__":
    asyncio.run(main())
