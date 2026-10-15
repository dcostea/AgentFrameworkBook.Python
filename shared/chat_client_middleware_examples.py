"""Chapter 09 policies at the chat-client and function-invocation layers."""

import copy
from datetime import datetime, timezone
import re
import threading
from time import perf_counter

from agent_framework import (
    Agent, ChatMiddleware, ChatMiddlewareLayer, ChatResponse, ChatResponseUpdate,
    Content, FunctionInvocationLayer, FunctionMiddleware, Message, ResponseStream,
)
from agent_framework.openai import RawOpenAIChatCompletionClient
from pydantic import BaseModel

from .chat_clients import create_completion_client, model_for
from .color_helper import Color, print_colored_line
from .tool_examples import create_motor_tools

MOTOR_INSTRUCTIONS = """You are the MotorsAgent controlling a robot car.
Break down complex movement commands into basic moves: forward, backward, turn left, turn right, stop.
Respond with the sequence of moves needed to accomplish the task."""
JSON_INSTRUCTIONS = """You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Use a JSON array like [move1, move2, move3] for the response.
Respond only with the moves and their parameters (angle or distance), without any additional explanations."""
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
EMAIL_MASK = "[REDACTED-EMAIL]"


class LimitExceededException(RuntimeError):
    def __init__(self, current_count: int, max_requests: int):
        self.current_count = current_count
        self.max_requests = max_requests
        super().__init__(f"Request limit exceeded: request {current_count} of {max_requests} blocked!")


class LimitRequests(ChatMiddleware):
    def __init__(self, max_requests: int = 2):
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        self.max_requests = max_requests
        self.request_count = 0
        self._lock = threading.Lock()

    async def process(self, context, call_next):
        with self._lock:
            self.request_count += 1
            count = self.request_count
        print_colored_line(f"[ChatClient] [SharedFunction] [Limit] PRE: Request {count} of {self.max_requests} max", Color.YELLOW)
        if count > self.max_requests:
            raise LimitExceededException(count, self.max_requests)
        await call_next()

        def completed():
            print_colored_line(f"[ChatClient] [SharedFunction] [Limit] POST: Request {count} completed", Color.YELLOW)

        if context.stream:
            context.stream_cleanup_hooks.append(completed)
        else:
            completed()


class RemoveEmail(ChatMiddleware):
    async def process(self, context, call_next):
        print_colored_line(f"[ChatClient] [SharedFunction] [Email] PRE: Scanning {len(context.messages)} messages...", Color.YELLOW)
        sanitized = []
        found = False
        for message in context.messages:
            if message.role != "user":
                sanitized.append(message)
                continue
            clone = copy.copy(message)
            clone.contents = []
            for content in message.contents:
                if content.type == "text":
                    clean = EMAIL_PATTERN.sub(EMAIL_MASK, content.text)
                    found = found or clean != content.text
                    new_content = copy.copy(content)
                    new_content.text = clean
                    clone.contents.append(new_content)
                else:
                    clone.contents.append(content)
            sanitized.append(clone)
        context.messages = sanitized
        print_colored_line(f"[ChatClient] [SharedFunction] [Email] {'Email detected and removed!' if found else 'No email found'}", Color.YELLOW)
        await call_next()

        def completed():
            print_colored_line("[ChatClient] [SharedFunction] [Email] POST: Completed", Color.YELLOW)

        if context.stream:
            context.stream_cleanup_hooks.append(completed)
        else:
            completed()


class EnforceTokenBudget(ChatMiddleware):
    def __init__(self, max_tokens: int = 2000):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.tokens_count = 0
        self._lock = threading.Lock()

    def record_usage(self, response):
        usage = response.usage_details or {}
        total = usage.get("total_token_count")
        if total is None:
            total = (usage.get("input_token_count") or 0) + (usage.get("output_token_count") or 0)
        with self._lock:
            self.tokens_count += total
        return response

    async def process(self, context, call_next):
        with self._lock:
            exhausted = self.tokens_count >= self.max_tokens
        if exhausted:
            print_colored_line(f"[ChatClient] [Response] [Tokens] Budget exhausted ({self.tokens_count} / {self.max_tokens}) — LLM call skipped", Color.RED)
            message = Message("assistant", ["Token budget exhausted."])
            if context.stream:
                async def synthetic():
                    yield ChatResponseUpdate(role="assistant", contents=message.contents)
                context.result = ResponseStream(synthetic(), finalizer=lambda _: ChatResponse(messages=[message]))
            else:
                context.result = ChatResponse(messages=[message])
            return
        await call_next()
        if context.stream:
            context.stream_result_hooks.append(self.record_usage)
        else:
            self.record_usage(context.result)


class AddTimestamp(ChatMiddleware):
    def __init__(self, clock=None):
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def process(self, context, call_next):
        await call_next()
        timestamp = self.clock().isoformat()
        print_colored_line(f"[ChatClient] [Response] [Timestamp] Stamping response with [{timestamp}]", Color.CYAN)
        if context.stream:
            stamped_messages = set()

            def stamp(update):
                if update.role not in {None, "assistant"}:
                    return update
                key = update.message_id or update.response_id or "assistant"
                if key in stamped_messages:
                    return update
                for index, content in enumerate(update.contents):
                    if content.type == "text" and content.text:
                        clone = copy.copy(update)
                        clone.contents = list(update.contents)
                        text = copy.copy(content)
                        text.text = f"[{timestamp}] {content.text}"
                        clone.contents[index] = text
                        stamped_messages.add(key)
                        return clone
                return update

            context.stream_transform_hooks.append(stamp)
        else:
            for message in context.result.messages:
                if message.role != "assistant":
                    continue
                stamped = False
                for index, content in enumerate(message.contents):
                    if content.type == "text" and content.text and not stamped:
                        text = copy.copy(content)
                        text.text = f"[{timestamp}] {content.text}"
                        message.contents[index] = text
                        stamped = True


class ConstrainDistance(FunctionMiddleware):
    async def process(self, context, call_next):
        arguments = context.arguments.model_dump() if isinstance(context.arguments, BaseModel) else dict(context.arguments)
        if "backward" in context.function.name.lower() and "distance" in arguments:
            distance = int(arguments["distance"])
            if distance > 5:
                arguments["distance"] = 5
                context.arguments = arguments
                print_colored_line(f"[ChatClient] [FunctionCall] [Constrain] Backward distance constrained from {distance}m to 5m", Color.YELLOW)
        await call_next()


class AuditFunctionCalling(FunctionMiddleware):
    async def process(self, context, call_next):
        started = perf_counter()
        timestamp = datetime.now(timezone.utc)
        await call_next()
        print_colored_line(
            f"[ChatClient] [FunctionCall] [Audit] Function call '{context.function.name}' "
            f"[{timestamp:%H:%M:%S}] COMPLETED in {(perf_counter()-started)*1000:.0f}ms | Result: {context.result}",
            Color.GREEN,
        )


class ExternalCallChatClient(ChatMiddlewareLayer, FunctionInvocationLayer, RawOpenAIChatCompletionClient):
    """Place chat policies outside the function loop, matching the mixed C# builder."""

    def __init__(self, sdk, *, provider="openai", chat_middleware=(), function_middleware=()):
        super().__init__(model=model_for(provider), async_client=sdk)
        self.chat_middleware = list(chat_middleware)
        self.function_middleware = list(function_middleware)


PROMPTS = {
    "baseline": [
        "Advance 100 meters, turn right and go 9 meters, then contact me at john.doe@example.com for the updates.",
        "Move forward 3 meters, turn right 90 degrees, move forward 3 meters",
        "Move forward 10 meters then go backward 10 meters",
    ],
    "shared": [
        "There is a tree directly in front of the car. Avoid it and then return to the original path.",
        "Navigate to original position. Contact me at john.doe@example.com for updates.",
        "Stop and email me at sara.doe@example.com for further instructions.",
    ],
    "response": ["Move forward 3 meters", "Turn right 45 degrees"],
    "function": ["Move forward 10 meters then go backward 8 meters"],
    "mixed": [
        "Navigate to original position. Contact me at john.doe@example.com for updates.",
        "Move forward 10 meters then go backward 10 meters",
        "Move forward 3 meters, turn right 90 degrees, move forward 3 meters",
    ],
}


def build_demo(sdk, kind: str, *, provider="openai") -> Agent:
    if kind not in PROMPTS:
        raise ValueError(f"Unknown middleware demo: {kind}")
    shared = [LimitRequests(), RemoveEmail()] if kind in {"shared", "mixed"} else []
    response = [EnforceTokenBudget(), AddTimestamp()] if kind in {"response", "mixed"} else []
    functions = [ConstrainDistance(), AuditFunctionCalling()] if kind in {"function", "mixed"} else []
    if kind == "mixed":
        client = ExternalCallChatClient(sdk, provider=provider, chat_middleware=shared + response, function_middleware=functions)
    else:
        client = create_completion_client(sdk, provider, middleware=shared + response + functions)
    return Agent(client=client, name="MotorsAgent", description="Controls robot car movements.",
                 instructions=JSON_INSTRUCTIONS if kind == "shared" else MOTOR_INSTRUCTIONS,
                 tools=[] if kind == "shared" else create_motor_tools())


async def run_demo(agent, kind: str, *, stream=False):
    session = agent.create_session()
    results = []
    for prompt in PROMPTS[kind]:
        print_colored_line(f"QUERY: {prompt}", Color.YELLOW)
        try:
            if stream:
                response_stream = agent.run(prompt, session=session, stream=True)
                print("RESULT: ", end="", flush=True)
                async for update in response_stream:
                    if update.text:
                        print(update.text, end="", flush=True)
                result = await response_stream.get_final_response()
                print()
            else:
                result = await agent.run(prompt, session=session)
                print_colored_line(f"RESULT: {result.text}\n", Color.GREEN)
            results.append(result)
        except LimitExceededException as exc:
            print_colored_line(f"EXCEPTION: {exc}\n", Color.RED)
    return results
