"""Chapter 10: native Agent Framework 1.8.1 run and function middleware."""

import argparse
import asyncio
import contextlib
import contextvars
import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_framework import (
    Agent, AgentContext, AgentMiddleware, AgentResponse, AgentResponseUpdate,
    ChatContext, ChatMiddleware, Content, FunctionInvocationContext, FunctionMiddleware,
    Message, ResponseStream,
)

from .chat_clients import create_completion_client, create_sdk_client
from .tool_examples import INSTRUCTIONS, create_motor_tools

MOTORS_INSTRUCTIONS = """You are the MotorsAgent controlling a robot car.
Break down complex movement commands into basic moves: forward, backward, turn_left, turn_right, stop.
Respond with the sequence of moves needed to accomplish the task."""
MOTORS_DESCRIPTION = "Controls robot car movements: forward, backward, turn left/right, stop."
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
EMAIL_MASK = "[REDACTED-EMAIL]"
BLOCKED_SEQUENCE = "BLOCKED: cannot execute the command due to illegal reversal. Issue a stop command first."
PROJECTS = (
    "AgentFunctionCallingMiddleware", "AgentMiddlewareMixed", "AgentResponseMiddleware",
    "AgentSharedFunctionMiddleware", "AgentWithLogging",
)


class OperationDeniedException(Exception):
    def __init__(self, operator_name: str):
        super().__init__(
            f"Operation denied: operator '{operator_name}' is not authorised to issue commands. "
            "Only 'driver' role is permitted."
        )


def _safe(value: Any) -> str:
    text = EMAIL_PATTERN.sub(EMAIL_MASK, str(value))
    text = re.sub(r"\bsk-[\w-]+", "[REDACTED-KEY]", text)
    return re.sub(r"(?i)((?:api[_-]?key|password|token)\s*[=:]\s*)\S+", r"\1[REDACTED]", text)


def _calls(messages):
    return [content for message in messages for content in message.contents if content.type == "function_call"]


def _replace_text(message: Message, text: str) -> Message:
    # C# replaces Contents wholesale; retain tool calls, media and message metadata in Python.
    result = copy.copy(message)
    contents = []
    inserted = False
    for content in message.contents:
        if content.type == "text":
            if not inserted:
                replacement = copy.copy(content)
                replacement.text = text
                contents.append(replacement)
                inserted = True
        else:
            contents.append(content)
    if not inserted:
        contents.append(Content.from_text(text))
    result.contents = contents
    return result


class ErrorLoggingResponseStream(ResponseStream):
    """Observe ordinary stream errors without extending native cancellation semantics."""

    def __init__(self, source, error_hooks):
        self.source = source
        self.error_hooks = error_hooks
        super().__init__(source, finalizer=self._finalize)

    async def _finalize(self, updates):
        return await self.source.get_final_response()

    async def __anext__(self):
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            raise
        except Exception as error:
            for hook in self.error_hooks:
                hook(error)
            raise


@dataclass
class InvocationState:
    agent_name: str
    redact_logs: bool = False
    messages: list[Message] = field(default_factory=list)
    pending: list[Content] = field(default_factory=list)


_invocation = contextvars.ContextVar("chapter10_invocation", default=None)


@contextlib.contextmanager
def _activate(state):
    token = _invocation.set(state)
    try:
        yield
    finally:
        _invocation.reset(token)


class FrameworkFailureFilter(logging.Filter):
    def filter(self, record):
        state = _invocation.get()
        if state is not None and state.redact_logs:
            message = record.getMessage()
            if "Function failed. Error:" in message or "raised an exception; returning an error result" in message:
                record.msg = "Function failed; exception details omitted from logs."
                record.args = ()
                record.exc_info = None
                record.exc_text = None
        return True


_framework_failure_filter = FrameworkFailureFilter()


class InvocationScope(AgentMiddleware):
    def __init__(self, *, redact_logs=False):
        self.redact_logs = redact_logs

    async def process(self, context: AgentContext, call_next):
        state = InvocationState(context.agent.name, redact_logs=self.redact_logs)
        with _activate(state):
            await call_next()
        if isinstance(context.result, AgentResponse) and context.result.finish_reason is None:
            context.result.finish_reason = getattr(context.result.raw_representation, "finish_reason", None)
        if context.stream and isinstance(context.result, ResponseStream):
            context.result.with_pull_context_manager(lambda: _activate(state))
            error_hooks = context.metadata.get("stream_error_hooks", ())
            if error_hooks:
                context.result = ErrorLoggingResponseStream(context.result, error_hooks)


class InvocationTranscript(ChatMiddleware):
    """Bridge the transcript absent from Python FunctionInvocationContext."""

    async def process(self, context: ChatContext, call_next):
        state = _invocation.get()
        if state is None:
            await call_next()
            return
        state.messages = list(context.messages)
        state.pending = []

        def record(response):
            state.pending = _calls(response.messages)
            return response

        await call_next()
        if context.stream:
            context.stream_result_hooks.append(record)
        else:
            record(context.result)


class AgentGuardrails(AgentMiddleware):
    def __init__(self, *, mixed=False, emit=print):
        self.mixed = mixed
        self.emit = emit

    async def process(self, context: AgentContext, call_next):
        state = context.session.state if context.session else {}
        operator = state.get("OperatorName")
        operator = operator if operator is not None else ("none" if self.mixed else "unknown")
        mission = state.get("MissionTag")
        prefix = "[Agent] [SharedFunction] [Guardrails]"
        self.emit(f"{prefix} PRE: operator='{operator}', mission='{mission if mission is not None else 'none'}'")
        allowed = operator == "driver" if self.mixed else operator.lower() == "driver"
        if not allowed:
            raise OperationDeniedException(operator)
        if mission is not None:
            latest = _replace_text(context.messages[-1], f"[{mission}] {context.messages[-1].text}")
            latest.role = "user"
            context.messages = [*context.messages[:-1], latest]
        self.emit(f"{prefix} Mission tag: '{mission if mission is not None else 'none'}'")
        await call_next()

        def post(response):
            self.emit(f"{prefix} POST: Completed for operator='{operator}', mission='{mission if mission is not None else 'none'}'")
            return response

        if context.stream:
            context.stream_result_hooks.append(post)
            context.stream_cleanup_hooks.append(lambda: self.emit(f"{prefix} CLEANUP"))
        else:
            post(context.result)


class PersistentRemoveEmail(AgentMiddleware):
    def __init__(self, *, emit=print):
        self.emit = emit

    async def process(self, context: AgentContext, call_next):
        prefix = "[Agent] [SharedFunction] [Email]"
        self.emit(f"{prefix} PRE: Scanning {len(context.messages)} messages for email...")
        found = False
        messages = []
        for message in context.messages:
            if message.role == "user":
                sanitized = EMAIL_PATTERN.sub(EMAIL_MASK, message.text)
                found |= sanitized != message.text
                messages.append(_replace_text(message, sanitized))
            else:
                messages.append(message)
        context.messages = messages
        self.emit(f"{prefix} {'Email detected and removed!' if found else 'No email found'}")
        await call_next()

        def post(response):
            self.emit(f"{prefix} POST: Completed")
            return response

        if context.stream:
            context.stream_result_hooks.append(post)
            context.stream_cleanup_hooks.append(lambda: self.emit(f"{prefix} CLEANUP"))
        else:
            post(context.result)


def _buffered_stream(response: AgentResponse):
    async def updates():
        for message in response.messages:
            yield AgentResponseUpdate(
                contents=message.contents, role=message.role, author_name=message.author_name,
                message_id=message.message_id, response_id=response.response_id,
                created_at=response.created_at, finish_reason=response.finish_reason,
                additional_properties=response.additional_properties,
            )
        if response.usage_details:
            yield AgentResponseUpdate(contents=[Content.from_usage(response.usage_details)])
    return ResponseStream(updates(), finalizer=lambda _: response)


class CompletedResponseMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        streaming = context.stream
        # C# runStreamingFunc:null also falls back to a complete, non-streaming run.
        context.stream = False
        try:
            await call_next()
            context.result = self.transform(context, context.result)
        finally:
            context.stream = streaming
        if context.result.finish_reason is None:
            context.result.finish_reason = getattr(context.result.raw_representation, "finish_reason", None)
        if streaming:
            context.result = _buffered_stream(context.result)


class CaptainsLog(CompletedResponseMiddleware):
    def __init__(self, *, emit=print, clock=None):
        self.emit = emit
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def transform(self, context, response):
        count = len(_calls(response.messages))
        tool_tag = f" Tools fired: {count}." if count else ""
        mission = context.session.state.get("MissionTag") if context.session else None
        prefix = f"Stardate {self.clock().isoformat()}. Mission {mission if mission is not None else ''}. Agent {context.agent.name}.{tool_tag} "
        self.emit(f"[Agent] [Response] [CaptainsLog] {prefix}")
        response.messages = [
            _replace_text(message, prefix + message.text) if message.role == "assistant" and message.text else message
            for message in response.messages
        ]
        return response


class MovementSequenceAuditor(CompletedResponseMiddleware):
    def __init__(self, *, emit=print):
        self.emit = emit

    def transform(self, context, response):
        sequence = "-".join(call.name.lower() for call in _calls(response.messages))
        if "forward-backward" in sequence or "backward-forward" in sequence:
            self.emit("[Agent] [Response] [SequenceAuditor] ILLEGAL sequence detected — stop required between direction reversals")
            for index in range(len(response.messages) - 1, -1, -1):
                message = response.messages[index]
                if message.role == "assistant":
                    warning = f"WARNING: illegal direction '{message.text}' reversal detected in executed sequence. A stop command was missing."
                    response.messages[index] = _replace_text(message, warning)
                    break
        return response


class PreventDangerousMoves(FunctionMiddleware):
    def __init__(self, *, mixed=False, emit=print):
        self.mixed = mixed
        self.emit = emit

    async def process(self, context: FunctionInvocationContext, call_next):
        state = _invocation.get()
        if state is None:
            raise RuntimeError("PreventDangerousMoves requires InvocationScope and InvocationTranscript")
        prefix = "[Agent] [FunctionCall] [Safety]"
        if state.agent_name != "MotorsAgent":
            self.emit(f"{prefix} SKIP: rule does not apply to agent '{state.agent_name}'")
            await call_next()
            return
        if self.mixed:
            sequence = "-".join(call.name.lower() for call in [*_calls(state.messages), *state.pending])
            illegal = "forward-backward" in sequence or "backward-forward" in sequence
            if illegal:
                self.emit(f"{prefix} BLOCKED: illegal reversal detected — a 'stop' is required first.")
                context.result = BLOCKED_SEQUENCE
                return
        else:
            prior = _calls(state.messages)
            for call in state.pending:
                if call.call_id == context.metadata.get("call_id"):
                    break
                prior.append(call)
            previous = prior[-1].name if prior else ""
            current = context.function.name
            if f"{previous}-{current}" in {"forward-backward", "backward-forward"}:
                self.emit(f"{prefix} BLOCKED: '{previous}-{current}' is an illegal reversal — a 'stop' is required first.")
                context.result = f"BLOCKED: cannot execute '{current}' directly after '{previous}'. Issue a stop command first."
                return
            self.emit(f"{prefix} SAFE: '{current}' (last: '{previous}')")
        await call_next()


class AuditAgentFunctionCalls(FunctionMiddleware):
    def __init__(self, *, emit=print):
        self.emit = emit

    async def process(self, context: FunctionInvocationContext, call_next):
        state = _invocation.get()
        name = state.agent_name if state else "unknown"
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        args = context.arguments.model_dump() if hasattr(context.arguments, "model_dump") else context.arguments
        args_text = ", ".join(f"{key}={_safe(value)}" for key, value in args.items())
        prefix = f"[Agent] [FunctionCall] [Audit] [{timestamp}]"
        self.emit(f"{prefix} Agent '{name}' invoking '{context.function.name}({args_text})'")
        start = time.perf_counter()
        try:
            await call_next()
            self.emit(f"{prefix} COMPLETED | Result: {_safe(context.result)} | Elapsed: {(time.perf_counter() - start) * 1000:.3f}ms")
        except Exception as error:
            self.emit(f"{prefix} FAILED | Error: {_safe(error)} | Elapsed: {(time.perf_counter() - start) * 1000:.3f}ms")
            raise


class StructuredAgentLogging(AgentMiddleware):
    def __init__(self, logger):
        self.logger = logger

    async def process(self, context: AgentContext, call_next):
        start = time.perf_counter()
        metadata = {"agent_name": context.agent.name, "operation": "agent.run", "streaming": context.stream}
        self.logger.debug("Agent run started", extra={**metadata, "message_count": len(context.messages)})

        def completed(response):
            if response.finish_reason is None:
                response.finish_reason = getattr(response.raw_representation, "finish_reason", None)
            self.logger.info("Agent run completed", extra={
                **metadata, "elapsed_ms": (time.perf_counter() - start) * 1000,
                "tool_call_count": len(_calls(response.messages)), "finish_reason": str(response.finish_reason),
                "usage_details": dict(response.usage_details or {}), "response_id": response.response_id,
            })
            return response

        def failed(error):
            self.logger.error("Agent run failed", extra={
                **metadata, "elapsed_ms": (time.perf_counter() - start) * 1000,
                "exception_type": type(error).__name__,
            })

        try:
            await call_next()
            if context.stream:
                context.stream_result_hooks.append(completed)
                context.stream_cleanup_hooks.append(lambda: self.logger.debug("Agent stream cleanup", extra=metadata))
                context.metadata.setdefault("stream_error_hooks", []).append(failed)
            else:
                completed(context.result)
        except Exception as error:
            failed(error)
            raise


class StructuredFunctionLogging(FunctionMiddleware):
    def __init__(self, logger):
        self.logger = logger

    async def process(self, context: FunctionInvocationContext, call_next):
        state = _invocation.get()
        metadata = {"agent_name": state.agent_name if state else "unknown", "operation": "tool.invoke",
                    "tool_name": context.function.name, "call_id": context.metadata.get("call_id")}
        start = time.perf_counter()
        self.logger.debug("Tool invocation started", extra=metadata)
        try:
            await call_next()
            self.logger.info("Tool invocation completed", extra={**metadata, "elapsed_ms": (time.perf_counter() - start) * 1000})
        except Exception as error:
            self.logger.error("Tool invocation failed", extra={
                **metadata, "elapsed_ms": (time.perf_counter() - start) * 1000,
                "exception_type": type(error).__name__,
            })
            raise


def build_agent(sdk, project: str, *, provider="openai", tools=None, emit=print, logger=None) -> Agent:
    if project not in PROJECTS:
        raise ValueError(f"Unknown chapter 10 project: {project}")
    client = create_completion_client(sdk, provider)
    logging_project = project == "AgentWithLogging"
    if logging_project:
        logging.getLogger("agent_framework").addFilter(_framework_failure_filter)
    middleware = [InvocationScope(redact_logs=logging_project)]
    if project in {"AgentSharedFunctionMiddleware", "AgentMiddlewareMixed"}:
        middleware += [AgentGuardrails(mixed=project == "AgentMiddlewareMixed", emit=emit), PersistentRemoveEmail(emit=emit)]
    if project in {"AgentResponseMiddleware", "AgentMiddlewareMixed"}:
        middleware += [MovementSequenceAuditor(emit=emit), CaptainsLog(emit=emit)]
    if project in {"AgentFunctionCallingMiddleware", "AgentMiddlewareMixed"}:
        middleware += [PreventDangerousMoves(mixed=project == "AgentMiddlewareMixed", emit=emit), AuditAgentFunctionCalls(emit=emit), InvocationTranscript()]
    if logging_project:
        logger = logger or logging.getLogger("chapter10.AgentWithLogging")
        middleware += [StructuredAgentLogging(logger), StructuredFunctionLogging(logger)]
    return Agent(
        client=client, name="RobotCarAgent" if logging_project else "MotorsAgent",
        description="An agent that assists a robot with the basic moves." if logging_project else MOTORS_DESCRIPTION,
        instructions=INSTRUCTIONS if logging_project else MOTORS_INSTRUCTIONS,
        tools=create_motor_tools() if tools is None else tools, middleware=middleware,
    )


@dataclass(frozen=True)
class Demo:
    scenario: str
    prompt: str
    session: int = 1
    operator: str | None = None
    mission: str | None = None


DEMOS = {
    "AgentFunctionCallingMiddleware": (
        Demo("safe", "Move forward 10 meters."),
        Demo("reversal", "Move forward 2 meters and immediatelly move backward 2 meters."),
    ),
    "AgentResponseMiddleware": (
        Demo("captain", "Move forward 5 meters", mission="MISSION-45 | EXPLORATION"),
        Demo("reversal", "Move forward 2 meters and immediatelly move backward 2 meters."),
    ),
    "AgentSharedFunctionMiddleware": (
        Demo("authorized", "Navigate forward 5 meters. My email is john.doe@example.com if needed.", operator="driver", mission="MISSION-42 | PERIMETER-SCAN"),
        Demo("second", "Turn left 90 degrees."),
        Demo("denied", "Move forward 3 meters.", operator="observer"),
    ),
    "AgentMiddlewareMixed": (
        Demo("authorized", "Navigate to original position. Contact me at john.doe@example.com for updates.", operator="driver", mission="MISSION-42 | PERIMETER-SCAN"),
        Demo("reversal", "Move forward 10 meters then go backward 10 meters"),
        Demo("denied", "Move forward 3 meters", session=2, operator="observer", mission="MISSION-99 | INSPECTION"),
    ),
    "AgentWithLogging": (Demo("round", "go round."),),
}


async def run_demos(project, *, provider="openai", stream=False, scenario="default", sdk=None):
    owns_sdk = sdk is None
    sdk = sdk or create_sdk_client(provider)
    try:
        agent = build_agent(sdk, project, provider=provider)
        sessions = {}
        for index, demo in enumerate(DEMOS[project], 1):
            if demo.session not in sessions:
                sessions[demo.session] = agent.create_session()
            session = sessions[demo.session]
            if demo.operator is not None:
                session.state["OperatorName"] = demo.operator
            if demo.mission is not None:
                session.state["MissionTag"] = demo.mission
            if scenario != "default" and scenario != demo.scenario:
                continue
            print(f"--- TEST {index}: {demo.scenario} ---\nQUERY: {demo.prompt}")
            try:
                if stream:
                    response_stream = agent.run(demo.prompt, session=session, stream=True)
                    async for update in response_stream:
                        if update.text:
                            print(update.text, end="", flush=True)
                    response = await response_stream.get_final_response()
                    print()
                else:
                    response = await agent.run(demo.prompt, session=session)
                print(f"\nRESULT: {response}\n")
            except OperationDeniedException as error:
                print(f"EXCEPTION: {error}\n")
        if project == "AgentWithLogging":
            print(agent.name)
    finally:
        if owns_sdk:
            await sdk.close()


class StructuredLogFormatter(logging.Formatter):
    def format(self, record):
        fields = {key: getattr(record, key) for key in (
            "agent_name", "operation", "streaming", "message_count", "elapsed_ms", "tool_call_count",
            "finish_reason", "usage_details", "response_id", "tool_name", "call_id", "exception_type",
        ) if hasattr(record, key)}
        return f"[{self.formatTime(record, '%H:%M:%S')}] [{record.levelname}] {record.getMessage()} {json.dumps(fields, default=str)}"


def main_for(project: str, argv=None):
    parser = argparse.ArgumentParser(description=f"Chapter 10 — {project}")
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    parser.add_argument("--stream", action="store_true", help="Stream when supported; response middleware buffers as in C#.")
    parser.add_argument("--scenario", choices=["default", *(demo.scenario for demo in DEMOS[project])], default="default")
    args = parser.parse_args(argv)
    if project == "AgentWithLogging":
        logger = logging.getLogger("chapter10.AgentWithLogging")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(StructuredLogFormatter())
            logger.addHandler(handler)
        logger.propagate = False
    print(f"Agent Middleware — {project}")
    asyncio.run(run_demos(project, provider=args.provider, stream=args.stream, scenario=args.scenario))
