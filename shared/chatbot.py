"""Conversation loop shared by the four chapter 03 chatbots."""

from collections.abc import Callable, Mapping
from typing import Any

from agent_framework import Message, SlidingWindowStrategy, tool
from pydantic import BaseModel, Field

from . import motor_tools
from .chat_clients import ROBOT_SYSTEM


class StepsResponse(BaseModel):
    steps: list[str] = Field(alias="Steps")


def motor_functions() -> list:
    # These tools only print simulated moves; no hardware is controlled.
    return [tool(function, approval_mode="never_require") for function in motor_tools.as_ai_tools()]


async def run_chatbot(
    client: Any,
    *,
    system: str = ROBOT_SYSTEM,
    options: Mapping[str, Any] | None = None,
    stream: bool = False,
    reduce_history: bool = False,
    read_input: Callable[[str], str] = input,
) -> list[Message]:
    conversation = [Message("system", [system])]
    print(f"System:\n{system}\n")
    while True:
        try:
            command = read_input("User: ")
        except EOFError:
            break
        if not command:
            break
        prompt = f'## Context\nComplex command:\n"{command}"'
        conversation.append(Message("user", [prompt]))
        kwargs: dict[str, Any] = {"options": dict(options or {})}
        if reduce_history:
            # Keep tool-call/result groups intact instead of slicing individual messages.
            kwargs["compaction_strategy"] = SlidingWindowStrategy(keep_last_groups=10)
        if stream:
            response_stream = client.get_response(conversation, stream=True, **kwargs)
            print("\nAssistant: ", end="", flush=True)
            async for update in response_stream:
                if update.text:
                    print(update.text, end="", flush=True)
            response = await response_stream.get_final_response()
            print()
        else:
            response = await client.get_response(conversation, **kwargs)
            print(f"\nAssistant: {response.text}")
        conversation.extend(response.messages)
    return conversation
