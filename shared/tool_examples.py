"""Simulated motor tools and invocation loops for chapter 06."""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_framework import Agent, Content, Message, tool

from . import motor_tools
from .chat_clients import create_completion_client
from .color_helper import Color, print_colored_line

INSTRUCTIONS = """You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Respond only with the moves and their parameters (angle or distance), without any additional explanations."""
OBSTACLE_PROMPT = """Complex command:
"There is a tree directly in front of the car. Avoid it and then return to the original path."
"""
APPROVAL_TOOLS = frozenset({"backward", "stop"})


def create_motor_tools(*, require_approval: bool = False) -> list:
    return [
        tool(
            function,
            approval_mode="always_require" if require_approval and function.__name__ in APPROVAL_TOOLS else "never_require",
        )
        for function in motor_tools.as_ai_tools()
    ]


def build_agent(sdk, *, tools=None, manual: bool = False, options: Mapping[str, Any] | None = None,
                instructions: str = INSTRUCTIONS) -> Agent:
    client = create_completion_client(sdk)
    if manual:
        client.function_invocation_configuration["enabled"] = False
    return Agent(
        client=client, name="RobotCarAgent", description="An agent that assists a robot with the basic moves.",
        instructions=instructions, tools=tools if tools is not None else create_motor_tools(),
        default_options=dict(options or {}),
    )


def print_tools(messages: Sequence[Message]) -> None:
    for message in messages:
        for content in message.contents:
            if content.type == "text":
                print_colored_line(f"ASST RESP: {content.text}", Color.YELLOW)
            elif content.type == "function_call":
                print_colored_line(
                    f"TOOL CALL {content.call_id}: {content.name} {json.dumps(content.arguments)}", Color.CYAN,
                )
            elif content.type == "function_result":
                print_colored_line(f"TOOL RESP {content.call_id}: {content.result}", Color.BLUE)


async def run_manual_tools(agent: Agent, prompt: str, tools: Sequence, *, max_rounds: int = 20):
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    tool_map = {function.name: function for function in tools}
    session = agent.create_session()
    response = await agent.run(prompt, session=session)
    for _ in range(max_rounds):
        calls = [content for message in response.messages if message.role == "assistant"
                 for content in message.contents if content.type == "function_call"]
        if not calls:
            return response
        print(f"TOOL CALLS THIS ROUND: {len(calls)}")

        async def invoke(call: Content) -> Message:
            if call.name not in tool_map:
                raise RuntimeError(f"Requested unknown motor tool: {call.name}")
            arguments = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
            if arguments is not None and not isinstance(arguments, Mapping):
                raise ValueError(f"Tool arguments must be a JSON object: {call.name}")
            result = await tool_map[call.name].invoke(arguments=arguments, skip_parsing=True, tool_call_id=call.call_id)
            print(f"  Tool: {call.name} => {result}")
            return Message("tool", [Content.from_function_result(call.call_id, result=result)])

        results = await asyncio.gather(*(invoke(call) for call in calls))
        response = await agent.run(results, session=session)
    if any(content.type == "function_call" for message in response.messages for content in message.contents):
        raise RuntimeError(f"Manual tool invocation exceeded {max_rounds} rounds")
    return response


def prompt_for_approval(request: Content, read_input: Callable[[str], str] = input) -> bool:
    call = request.function_call
    try:
        answer = read_input(f"Agent invoking {call.name} {json.dumps(call.arguments)}. Approve? [Y/n] ")
    except EOFError:
        answer = "n"
    approved = answer.strip().lower() in {"", "y", "yes"}
    print(f"AI Tool '{call.name}': {'Approved' if approved else 'Denied'}")
    return approved


async def run_with_approval(agent: Agent, prompt: str, *, read_input: Callable[[str], str] = input,
                            max_rounds: int = 20):
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    session = agent.create_session()
    response = await agent.run(prompt, session=session)
    for _ in range(max_rounds):
        requests = [content for message in response.messages for content in message.contents
                    if content.type == "function_approval_request"]
        if not requests:
            return response
        approvals = []
        for request in requests:
            # Python wraps the entire batch if any tool needs approval; gate only the protected moves.
            approved = prompt_for_approval(request, read_input) if request.function_call.name in APPROVAL_TOOLS else True
            approvals.append(Message("user", [request.to_function_approval_response(approved)]))
        response = await agent.run(approvals, session=session)
    if any(content.type == "function_approval_request" for message in response.messages for content in message.contents):
        raise RuntimeError(f"Tool approval exceeded {max_rounds} rounds")
    return response
