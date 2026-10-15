"""Offline integration tests against real Agent + AsyncOpenAI + MockTransport."""

import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
import socket
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
    from shared import agent_middleware_examples as examples
from agent_framework import AgentMiddleware, AgentResponse, Content, Message, ResponseStream, tool

SENTINEL = "middleware-sentinel-42"
ERROR = "motor-error-sentinel-42"


def completion(text=SENTINEL, calls=(), *, finish=None):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": f"call_{index}_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for index, (name, args) in enumerate(calls)
        ]
    return {"id": "chatcmpl-sentinel", "object": "chat.completion", "created": 1700000000, "model": "offline-model",
            "choices": [{"index": 0, "message": message, "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}


def sse(payload):
    choice = payload["choices"][0]
    message = choice["message"]
    chunks = [{"role": "assistant"}]
    if message.get("tool_calls"):
        chunks.append({"tool_calls": [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]})
    if message.get("content"):
        text = message["content"]
        middle = len(text) // 2
        chunks += [{"content": text[:middle]}, {"content": text[middle:]}]
    rows = []
    base = {key: payload[key] for key in ("id", "created", "model")}
    for chunk in chunks:
        rows.append({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": chunk, "finish_reason": None}]})
    rows.append({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]})
    rows.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": payload["usage"]})
    return ("".join(f"data: {json.dumps(row)}\n\n" for row in rows) + "data: [DONE]\n\n").encode()


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_CHAT_MODEL_ID": "offline-model"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex", "dotenv.load_dotenv"):
            guard = patch(target, side_effect=AssertionError(f"Offline test attempted {target}"))
            guard.start()
            self.addCleanup(guard.stop)
        self.output = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.output)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)
        self.requests = []
        self.events = []
        self.executed = []
        self.payloads = []
        self.logger = logging.getLogger(f"chapter10.test.{self.id()}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.records = Records()
        self.logger.addHandler(self.records)
        self.addCleanup(self.logger.removeHandler, self.records)
        console_logger = logging.getLogger("chapter10.AgentWithLogging")
        original = (console_logger.handlers[:], console_logger.level, console_logger.propagate)
        self.addCleanup(setattr, console_logger, "handlers", original[0])
        self.addCleanup(setattr, console_logger, "level", original[1])
        self.addCleanup(setattr, console_logger, "propagate", original[2])
        console_logger.handlers = [logging.NullHandler()]
        native_logger = logging.getLogger("agent_framework")
        native_filters = native_logger.filters[:]
        self.addCleanup(setattr, native_logger, "filters", native_filters)

        async def handler(request):
            body = json.loads(request.content)
            self.requests.append(body)
            self.assertEqual(request.url.host, "offline.invalid")
            payload = self.payloads.pop(0) if self.payloads else completion()
            if isinstance(payload, Exception):
                raise payload
            if body.get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(payload))
            return httpx.Response(200, json=payload)

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.sdk = AsyncOpenAI(api_key="offline-not-a-secret", base_url="https://offline.invalid/v1/", http_client=self.http, max_retries=0)
        self.addAsyncCleanup(self.sdk.close)

    def tools(self, *, failure=False, approval=False, error_message=ERROR):
        executed = self.executed

        @tool
        async def forward(distance: int) -> str:
            executed.append(("forward", distance))
            if failure:
                raise RuntimeError(error_message)
            return f"moved forward for {distance} meters."

        @tool(approval_mode="always_require" if approval else "never_require")
        async def backward(distance: int) -> str:
            executed.append(("backward", distance))
            return f"moved backward for {distance} meters."

        @tool
        async def stop() -> str:
            executed.append(("stop", None))
            return "stopped."

        @tool
        async def turn_left(angle: int) -> str:
            executed.append(("turn_left", angle))
            return f"turned anticlockwise {angle} degrees."

        return [forward, backward, stop, turn_left]

    def agent(self, project, **kwargs):
        return examples.build_agent(self.sdk, project, tools=kwargs.pop("tools", self.tools()), emit=self.events.append,
                                    logger=self.logger, **kwargs)

    def session(self, agent, operator="driver", mission="MISSION-42 | PERIMETER-SCAN"):
        session = agent.create_session()
        if operator is not None:
            session.state["OperatorName"] = operator
        if mission is not None:
            session.state["MissionTag"] = mission
        return session

    def tool_results(self, response):
        return [content for message in response.messages for content in message.contents if content.type == "function_result"]

    async def test_default_composed_order_and_agent_metadata(self):
        expected = {
            "AgentFunctionCallingMiddleware": ["PreventDangerousMoves", "AuditAgentFunctionCalls", "InvocationTranscript"],
            "AgentMiddlewareMixed": ["AgentGuardrails", "PersistentRemoveEmail", "MovementSequenceAuditor", "CaptainsLog", "PreventDangerousMoves", "AuditAgentFunctionCalls", "InvocationTranscript"],
            "AgentResponseMiddleware": ["MovementSequenceAuditor", "CaptainsLog"],
            "AgentSharedFunctionMiddleware": ["AgentGuardrails", "PersistentRemoveEmail"],
            "AgentWithLogging": ["StructuredAgentLogging", "StructuredFunctionLogging"],
        }
        for project, classes in expected.items():
            with self.subTest(project=project):
                agent = self.agent(project)
                self.assertEqual([type(m).__name__ for m in agent.middleware], ["InvocationScope", *classes])
                self.assertEqual(agent.name, "RobotCarAgent" if project == "AgentWithLogging" else "MotorsAgent")
                self.assertEqual(agent.default_options["instructions"], examples.INSTRUCTIONS if project == "AgentWithLogging" else examples.MOTORS_INSTRUCTIONS)

    async def test_validation_denial_prevents_network_tools_and_inner_middleware(self):
        for project, operators in [("AgentSharedFunctionMiddleware", [None, "observer"]),
                                   ("AgentMiddlewareMixed", [None, "observer", "Driver", "DRIVER"])]:
            for operator in operators:
                for stream in (False, True):
                    with self.subTest(project=project, operator=operator, stream=stream):
                        agent = self.agent(project)
                        session = self.session(agent, operator)
                        self.events.clear()
                        with self.assertRaisesRegex(examples.OperationDeniedException, "Only 'driver' role is permitted"):
                            result = agent.run("Move forward 3 meters", session=session, stream=stream)
                            if stream:
                                await result.get_final_response()
                            else:
                                await result
                        self.assertEqual(self.requests, [])
                        self.assertEqual(self.executed, [])
                        self.assertEqual(len(self.events), 1)
                        self.assertIn("operator='none'" if project.endswith("Mixed") and operator is None else
                                      "operator='unknown'" if operator is None else f"operator='{operator}'", self.events[0])

    async def test_standalone_gate_is_case_insensitive_not_whitespace_tolerant(self):
        agent = self.agent("AgentSharedFunctionMiddleware")
        for operator in ("driver", "Driver", "DRIVER"):
            with self.subTest(operator=operator):
                response = await agent.run("Move", session=self.session(agent, operator))
                self.assertEqual(response.text, SENTINEL)
        count = len(self.requests)
        with self.assertRaises(examples.OperationDeniedException):
            await agent.run("Move", session=self.session(agent, " driver "))
        self.assertEqual(len(self.requests), count)

    async def test_mission_email_before_mutation_and_persistent_history(self):
        agent = self.agent("AgentSharedFunctionMiddleware")
        session = self.session(agent)
        await agent.run("Navigate. Contact john.doe@example.com", session=session)
        await agent.run("Turn left", session=session)
        sent = self.requests[0]["messages"][-1]["content"]
        self.assertEqual(sent, "[MISSION-42 | PERIMETER-SCAN] Navigate. Contact [REDACTED-EMAIL]")
        second = json.dumps(self.requests[1])
        self.assertNotIn("john.doe@example.com", second)
        self.assertIn("[REDACTED-EMAIL]", second)
        self.assertIn("[MISSION-42 | PERIMETER-SCAN] Turn left", second)
        history = json.dumps(session.to_dict())
        self.assertNotIn("john.doe@example.com", history)
        phases = [event for event in self.events[:6] if "PRE:" in event or "POST:" in event]
        self.assertIn("[Guardrails] PRE:", phases[0])
        self.assertIn("[Email] PRE:", phases[1])
        self.assertIn("[Email] POST:", phases[2])
        self.assertIn("[Guardrails] POST:", phases[3])

    async def test_email_pattern_is_user_only_and_preserves_other_content(self):
        agent = self.agent("AgentSharedFunctionMiddleware")
        user = Message("user", [Content.from_text("Email a+b@example.com and c@example.X|z"),
                                Content.from_uri("https://offline.invalid/image.png", media_type="image/png")],
                       message_id="message-sentinel", additional_properties={"tag": SENTINEL})
        response = await agent.run([Message("system", ["system@example.com"]), user], session=self.session(agent, mission=None))
        self.assertEqual(response.text, SENTINEL)
        sent = json.dumps(self.requests[0])
        self.assertIn("system@example.com", sent)
        self.assertNotIn("a+b@example.com", sent)
        self.assertNotIn("c@example.X|", sent)
        self.assertIn("image_url", sent)
        self.assertEqual(user.text, "Email a+b@example.com and c@example.X|z")

    async def test_guardrail_prefixes_only_last_message_even_non_user(self):
        agent = self.agent("AgentSharedFunctionMiddleware")
        await agent.run([Message("user", ["first"]), Message("assistant", ["latest"])], session=self.session(agent))
        messages = self.requests[0]["messages"]
        self.assertEqual(messages[-2]["content"], "first")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "[MISSION-42 | PERIMETER-SCAN] latest")

    async def test_captain_concrete_result_edits_and_metadata(self):
        agent = self.agent("AgentResponseMiddleware")
        for middleware in agent.middleware:
            if isinstance(middleware, examples.CaptainsLog):
                middleware.clock = lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.payloads = [completion("", [("forward", {"distance": 5})]), completion("forward 5")]
        response = await agent.run("Move forward 5 meters", session=self.session(agent, mission="MISSION-45 | EXPLORATION"))
        self.assertEqual(response.text, "Stardate 2026-01-02T03:04:05+00:00. Mission MISSION-45 | EXPLORATION. Agent MotorsAgent. Tools fired: 1. forward 5")
        self.assertEqual(len(examples._calls(response.messages)), 1)
        self.assertEqual(len(self.tool_results(response)), 1)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage_details["total_token_count"], 36)
        self.assertEqual(response.response_id, "chatcmpl-sentinel")

    async def test_auditor_actual_warning_replaces_text_after_captain(self):
        agent = self.agent("AgentResponseMiddleware")
        self.payloads = [completion("", [("forward", {"distance": 2}), ("backward", {"distance": 2})]), completion("moves")]
        response = await agent.run("Reverse", session=self.session(agent))
        self.assertTrue(response.text.startswith("WARNING: illegal direction 'Stardate "))
        self.assertTrue(response.text.endswith("reversal detected in executed sequence. A stop command was missing."))
        self.assertEqual(len(examples._calls(response.messages)), 2)
        self.assertEqual(self.executed, [("forward", 2), ("backward", 2)])
        self.assertIn("[CaptainsLog]", self.events[0])
        self.assertIn("[SequenceAuditor]", self.events[1])

    async def test_auditor_normalizes_case_but_stop_or_turn_interrupts_pattern(self):
        for names, warning in [(["FORWARD", "BACKWARD"], True), (["forward", "stop", "backward"], False),
                               (["backward", "turn_left", "forward"], False)]:
            with self.subTest(names=names):
                # Manual tool mode still exercises the real transport and completed-run auditor.
                agent = self.agent("AgentResponseMiddleware")
                agent.client.function_invocation_configuration["enabled"] = False
                self.payloads = [completion("plan", [(name, {}) for name in names])]
                response = await agent.run("Plan")
                self.assertEqual(response.text.startswith("WARNING:"), warning)
                self.assertEqual([call.name for call in examples._calls(response.messages)], names)
                self.assertEqual([call.call_id for call in examples._calls(response.messages)],
                                 [f"call_{index}_{name}" for index, name in enumerate(names)])
                self.assertEqual(response.finish_reason, "tool_calls")
                self.assertEqual(response.usage_details["total_token_count"], 18)
                if not warning:
                    self.assertTrue(response.text.endswith("plan"))

    async def test_response_stream_matches_csharp_nonstream_fallback(self):
        agent = self.agent("AgentResponseMiddleware")
        self.payloads = [completion("", [("forward", {"distance": 2}), ("backward", {"distance": 2})]), completion(SENTINEL)]
        stream = agent.run("Reverse", session=self.session(agent), stream=True)
        updates = [update async for update in stream]
        response = await stream.get_final_response()
        self.assertTrue(response.text.startswith("WARNING:"))
        self.assertTrue(all(not request.get("stream") for request in self.requests))
        self.assertEqual("".join(update.text for update in updates), response.text)
        self.assertEqual(response.usage_details["total_token_count"], 36)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.response_id, "chatcmpl-sentinel")
        self.assertEqual(len(examples._calls(response.messages)), 2)

    async def test_standalone_safety_blocks_previous_current_and_skips_inner_audit(self):
        agent = self.agent("AgentFunctionCallingMiddleware")
        self.payloads = [completion("", [("forward", {"distance": 2}), ("backward", {"distance": 2})]), completion("replanned")]
        response = await agent.run("Reverse")
        self.assertEqual(self.executed, [("forward", 2)])
        results = self.tool_results(response)
        self.assertEqual(results[1].result, "BLOCKED: cannot execute 'backward' directly after 'forward'. Issue a stop command first.")
        self.assertEqual(sum("invoking" in event for event in self.events), 1)
        self.assertEqual(response.text, "replanned")

    async def test_standalone_stop_resets_rule_but_mixed_sequence_stays_illegal(self):
        for project, expected in [("AgentFunctionCallingMiddleware", [("forward", 2), ("stop", None), ("backward", 2)]),
                                  ("AgentMiddlewareMixed", [("forward", 2), ("stop", None), ("backward", 2)])]:
            with self.subTest(project=project):
                self.executed.clear()
                agent = self.agent(project)
                self.payloads = [completion("", [("forward", {"distance": 2}), ("stop", {}), ("backward", {"distance": 2})]), completion()]
                await agent.run("Stop before reversing", session=self.session(agent))
                self.assertEqual(self.executed, expected)
        agent = self.agent("AgentMiddlewareMixed")
        self.executed.clear()
        self.payloads = [completion("", [("forward", {"distance": 2}), ("backward", {"distance": 2})]),
                         completion("", [("stop", {})]), completion()]
        response = await agent.run("Reverse", session=self.session(agent))
        self.assertEqual(self.executed, [])
        self.assertEqual([result.result for result in self.tool_results(response)], [examples.BLOCKED_SEQUENCE] * 3)
        self.assertTrue(response.text.startswith("WARNING:"))

    async def test_mixed_full_sequence_short_circuit_blocks_entire_batch(self):
        agent = self.agent("AgentMiddlewareMixed")
        self.payloads = [completion("", [("forward", {"distance": 10}), ("backward", {"distance": 10})]), completion()]
        response = await agent.run("Contact john.doe@example.com", session=self.session(agent))
        self.assertEqual(self.executed, [])
        self.assertEqual(len(examples._calls(response.messages)), 2)
        self.assertEqual([result.result for result in self.tool_results(response)], [examples.BLOCKED_SEQUENCE] * 2)
        self.assertFalse(any("invoking" in event for event in self.events))
        labels = ["Guardrails] PRE", "Email] PRE", "Safety] BLOCKED", "CaptainsLog]", "SequenceAuditor]", "Email] POST", "Guardrails] POST"]
        positions = [next(i for i, event in enumerate(self.events) if label in event) for label in labels]
        self.assertEqual(positions, sorted(positions))

    async def test_safety_scoped_to_exact_agent_identity(self):
        for name in ("RobotCarAgent", "motorsagent"):
            with self.subTest(name=name):
                agent = self.agent("AgentFunctionCallingMiddleware")
                agent.name = name
                self.executed.clear()
                self.payloads = [completion("", [("forward", {"distance": 1}), ("backward", {"distance": 1})]), completion()]
                await agent.run("Reverse")
                self.assertEqual(self.executed, [("forward", 1), ("backward", 1)])
                self.assertTrue(any(f"SKIP: rule does not apply to agent '{name}'" in event for event in self.events))

    async def test_function_audit_exception_is_rethrown_then_framework_error_policy(self):
        agent = self.agent("AgentFunctionCallingMiddleware", tools=self.tools(failure=True))
        self.payloads = [completion("", [("forward", {"distance": 1})]), completion("failed move acknowledged")]
        response = await agent.run("Move")
        result = self.tool_results(response)[0]
        self.assertEqual(result.result, "Error: Function failed.")
        self.assertIn(ERROR, result.exception)
        self.assertTrue(any(f"FAILED | Error: {ERROR}" in event for event in self.events))
        self.assertFalse(any("COMPLETED" in event for event in self.events))
        self.assertEqual(response.text, "failed move acknowledged")

    async def test_invalid_function_arguments_do_not_reach_middleware_or_tools(self):
        agent = self.agent("AgentFunctionCallingMiddleware")
        self.payloads = [completion("", [("forward", {"distance": "invalid-distance-sentinel"})]), completion()]
        response = await agent.run("Move")
        self.assertEqual(self.executed, [])
        self.assertEqual(self.events, [])
        self.assertEqual(self.tool_results(response)[0].result, "Error: Argument parsing failed.")

    async def test_native_approval_is_not_bypassed(self):
        agent = self.agent("AgentFunctionCallingMiddleware", tools=self.tools(approval=True))
        self.payloads = [completion("", [("backward", {"distance": 1})])]
        session = self.session(agent)
        response = await agent.run("Backward", session=session)
        approvals = [content for message in response.messages for content in message.contents if content.type == "function_approval_request"]
        self.assertTrue(approvals)
        self.assertEqual(self.executed, [])
        self.assertFalse(any("invoking" in event for event in self.events))
        self.payloads = [completion("denied acknowledged")]
        denied = await agent.run([Message("user", [request.to_function_approval_response(False)]) for request in approvals], session=session)
        self.assertEqual(self.executed, [])
        self.assertEqual(denied.text, "denied acknowledged")

    async def test_native_stream_tool_execution_aggregation_and_safety(self):
        agent = self.agent("AgentFunctionCallingMiddleware")
        self.payloads = [completion("", [("forward", {"distance": 2}), ("backward", {"distance": 2})]), completion(SENTINEL, finish="length")]
        stream = agent.run("Reverse", stream=True)
        updates = [update async for update in stream]
        response = await stream.get_final_response()
        self.assertTrue(all(request["stream"] for request in self.requests))
        self.assertEqual(self.executed, [("forward", 2)])
        self.assertEqual(response.text, SENTINEL)
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.usage_details["total_token_count"], 36)
        self.assertEqual(len(examples._calls(response.messages)), 2)
        self.assertEqual(len(self.tool_results(response)), 2)
        self.assertEqual("".join(update.text for update in updates), SENTINEL)

    async def test_shared_stream_post_runs_after_consumption_once_and_preserves_usage(self):
        agent = self.agent("AgentSharedFunctionMiddleware")
        stream = agent.run("john.doe@example.com", stream=True, session=self.session(agent))
        await stream
        self.assertFalse(any("POST:" in event for event in self.events))
        response = await stream.get_final_response()
        await stream.get_final_response()
        self.assertEqual(response.text, SENTINEL)
        self.assertEqual(response.usage_details["total_token_count"], 18)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(sum("[Email] POST:" in event for event in self.events), 1)
        self.assertEqual(sum("[Guardrails] POST:" in event for event in self.events), 1)
        self.assertEqual(sum("CLEANUP" in event for event in self.events), 2)

    async def test_native_stream_cancelled_pull_and_pause_do_not_run_cleanup(self):
        for pause_before_cancel in (False, True):
            with self.subTest(pause_before_cancel=pause_before_cancel):
                started = asyncio.Event()
                release = asyncio.Event()
                body_cancelled = asyncio.Event()

                class PausedStream(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        try:
                            yield b'data: {"id":"paused","object":"chat.completion.chunk","created":1700000000,"model":"offline-model","choices":[{"index":0,"delta":{"role":"assistant","content":"first"},"finish_reason":null}]}\n\n'
                            started.set()
                            await release.wait()
                            yield b"data: [DONE]\n\n"
                        except asyncio.CancelledError:
                            body_cancelled.set()
                            raise

                async def handler(request):
                    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=PausedStream())

                sdk = AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/v1/", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
                self.addAsyncCleanup(sdk.close)
                agent = examples.build_agent(sdk, "AgentSharedFunctionMiddleware", tools=[], emit=self.events.append)
                self.events.clear()
                outer = agent.run("Move", stream=True, session=self.session(agent))
                await outer
                self.assertIsInstance(outer, ResponseStream)
                for unsupported in ("aclose", "__aenter__", "__aexit__"):
                    self.assertFalse(hasattr(outer, unsupported))
                self.assertEqual((await outer.__anext__()).text, "first")
                if pause_before_cancel:
                    # Stopping pulls is not cancellation and provides no abandoned-stream cleanup guarantee.
                    await asyncio.sleep(0)
                    self.assertFalse(started.is_set())
                    self.assertFalse(any("CLEANUP" in event for event in self.events))
                    self.assertFalse(any("POST:" in event for event in self.events))
                task = asyncio.create_task(outer.__anext__())
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(body_cancelled.is_set())
                self.assertFalse(any("CLEANUP" in event for event in self.events))
                self.assertFalse(any("POST:" in event for event in self.events))
                release.set()
                await sdk.close()

    async def test_stream_body_runtime_error_runs_native_cleanup_not_post(self):
        for project in ("AgentSharedFunctionMiddleware", "AgentWithLogging"):
            with self.subTest(project=project):
                class FailingBody(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        yield b'data: {"id":"failed-body","object":"chat.completion.chunk","created":1700000000,"model":"offline-model","choices":[{"index":0,"delta":{"role":"assistant","content":"first"},"finish_reason":null}]}\n\n'
                        raise RuntimeError(ERROR)

                async def handler(request):
                    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=FailingBody())

                sdk = AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/v1/", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
                self.addAsyncCleanup(sdk.close)
                agent = examples.build_agent(sdk, project, tools=[], emit=self.events.append, logger=self.logger)
                self.events.clear()
                self.records.records.clear()
                stream = agent.run("Move", stream=True, session=self.session(agent))
                self.assertEqual((await stream.__anext__()).text, "first")
                with self.assertRaises(Exception) as raised:
                    await stream.__anext__()
                self.assertIn(ERROR, str(raised.exception))
                self.assertFalse(any("POST:" in event for event in self.events))
                if project == "AgentSharedFunctionMiddleware":
                    self.assertEqual(sum("CLEANUP" in event for event in self.events), 2)
                else:
                    self.assertEqual(sum(record.getMessage() == "Agent stream cleanup" for record in self.records.records), 1)
                    self.assertEqual(sum(record.getMessage() == "Agent run failed" for record in self.records.records), 1)
                    self.assertFalse(any(record.getMessage() == "Agent run completed" for record in self.records.records))
                await sdk.close()

    async def test_logging_preserves_structured_metadata_timings_and_no_secrets(self):
        agent = self.agent("AgentWithLogging")
        self.payloads = [completion("", [("forward", {"distance": 5})]), completion(SENTINEL)]
        response = await agent.run("secret prompt sk-test-secret and john.doe@example.com")
        self.assertEqual(response.text, SENTINEL)
        completed = [record for record in self.records.records if record.getMessage() == "Agent run completed"][0]
        self.assertEqual(completed.agent_name, "RobotCarAgent")
        self.assertEqual(completed.tool_call_count, 1)
        self.assertEqual(completed.usage_details["total_token_count"], 36)
        self.assertEqual(completed.finish_reason, "stop")
        tool_record = [record for record in self.records.records if record.getMessage() == "Tool invocation completed"][0]
        self.assertEqual(tool_record.tool_name, "forward")
        self.assertGreaterEqual(tool_record.elapsed_ms, 0)
        logged = repr([record.__dict__ for record in self.records.records])
        self.assertNotIn("sk-test-secret", logged)
        self.assertNotIn("john.doe@example.com", logged)
        self.assertNotIn("secret prompt", logged)
        formatted = examples.StructuredLogFormatter().format(tool_record)
        self.assertIn('"tool_name": "forward"', formatted)
        self.assertIn('"agent_name": "RobotCarAgent"', formatted)

    async def test_logging_tool_and_transport_exception_policy(self):
        secret = "sensitive-exception-value-123"
        native_records = Records()
        native_logger = logging.getLogger("agent_framework")
        native_logger.addHandler(native_records)
        self.addCleanup(native_logger.removeHandler, native_records)
        agent = self.agent("AgentWithLogging", tools=self.tools(failure=True, error_message=f"{ERROR} api_key={secret}"))
        self.payloads = [completion("", [("forward", {"distance": 1})]), completion()]
        response = await agent.run("Move")
        self.assertTrue(native_records.records)
        self.assertNotIn(secret, repr([record.__dict__ for record in native_records.records]))
        self.assertIn(secret, self.tool_results(response)[0].exception)
        self.assertEqual(self.tool_results(response)[0].result, "Error: Function failed.")
        errors = [record for record in self.records.records if record.getMessage() == "Tool invocation failed"]
        self.assertEqual(errors[0].exception_type, "RuntimeError")
        self.assertNotIn(ERROR, repr(errors[0].__dict__))
        self.payloads = [httpx.ConnectError(ERROR)]
        with self.assertRaises(Exception):
            await agent.run("Move")
        self.assertTrue(any(record.getMessage() == "Agent run failed" and hasattr(record, "exception_type") for record in self.records.records))

    async def test_logging_stream_completion_is_deferred_and_not_duplicated(self):
        agent = self.agent("AgentWithLogging")
        stream = agent.run("go round.", stream=True)
        await stream
        self.assertFalse(any(record.getMessage() == "Agent run completed" for record in self.records.records))
        response = await stream.get_final_response()
        await stream.get_final_response()
        self.assertEqual(response.text, SENTINEL)
        self.assertEqual(sum(record.getMessage() == "Agent run completed" for record in self.records.records), 1)
        self.assertEqual(sum(record.getMessage() == "Agent stream cleanup" for record in self.records.records), 1)

    async def test_streaming_transport_failure_is_logged_and_cleaned_once(self):
        agent = self.agent("AgentWithLogging")
        self.payloads = [httpx.ConnectError(ERROR)]
        stream = agent.run("go round.", stream=True)
        with self.assertRaises(Exception):
            await stream.get_final_response()
        self.assertEqual(sum(record.getMessage() == "Agent run failed" for record in self.records.records), 1)
        self.assertEqual(sum(record.getMessage() == "Agent stream cleanup" for record in self.records.records), 1)
        self.assertFalse(any(record.getMessage() == "Agent run completed" for record in self.records.records))
        self.assertNotIn(ERROR, repr([record.__dict__ for record in self.records.records]))

    async def test_short_circuit_without_next_has_real_agent_result_no_network(self):
        class Cached(AgentMiddleware):
            async def process(self, context, call_next):
                context.result = AgentResponse(messages=[Message("assistant", ["cached-sentinel"])], finish_reason="stop")
        agent = self.agent("AgentResponseMiddleware")
        agent.middleware.append(Cached())
        response = await agent.run("Move")
        self.assertIn("cached-sentinel", response.text)
        self.assertTrue(response.text.startswith("Stardate "))
        self.assertEqual(self.requests, [])
        self.assertEqual(self.executed, [])

    async def test_all_five_entrypoints_help_cli_and_default_demos(self):
        for project in examples.PROJECTS:
            with self.subTest(project=project):
                path = ROOT / "10.AgentMiddleware" / project / "main.py"
                spec = importlib.util.spec_from_file_location(f"chapter10_{project}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with self.assertRaises(SystemExit) as exit_info:
                    module.main(["--help"])
                self.assertEqual(exit_info.exception.code, 0)
                with patch.object(examples, "run_demos", new_callable=Mock) as demos, patch.object(examples.asyncio, "run") as run:
                    module.main(["--provider", "azure", "--stream", "--scenario", examples.DEMOS[project][0].scenario])
                    demos.assert_called_once_with(project, provider="azure", stream=True, scenario=examples.DEMOS[project][0].scenario)
                    run.assert_called_once()
                for streaming in (False, True):
                    with self.subTest(streaming=streaming):
                        before = len(self.requests)
                        with patch("shared.motor_tools._DELAY_SECONDS", 0):
                            await examples.run_demos(project, sdk=self.sdk, stream=streaming)
                        expected = 1 if project == "AgentWithLogging" else 2
                        self.assertEqual(len(self.requests) - before, expected)
                        self.assertFalse(self.sdk.is_closed())
                        self.assertIn(examples.DEMOS[project][0].prompt, self.output.getvalue())

    async def test_concurrent_runs_do_not_share_inflight_transcript(self):
        async def handler(request):
            body = json.loads(request.content)
            await asyncio.sleep(0)
            prompt = next(message["content"] for message in body["messages"] if message["role"] == "user")
            if any(message["role"] == "tool" for message in body["messages"]):
                payload = completion(prompt)
            elif prompt == "reverse":
                payload = completion("", [("forward", {"distance": 2}), ("backward", {"distance": 2})])
            else:
                payload = completion("", [("backward", {"distance": 9})])
            if body.get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(payload))
            return httpx.Response(200, json=payload)

        sdk = AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/v1/", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        self.addAsyncCleanup(sdk.close)
        agent = examples.build_agent(sdk, "AgentFunctionCallingMiddleware", tools=self.tools(), emit=self.events.append)
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.executed.clear()

                async def run(prompt):
                    result = agent.run(prompt, stream=streaming, session=agent.create_session())
                    return await result.get_final_response() if streaming else await result

                reverse, independent = await asyncio.gather(run("reverse"), run("independent"))
                self.assertEqual(reverse.text, "reverse")
                self.assertEqual(independent.text, "independent")
                self.assertCountEqual(self.executed, [("forward", 2), ("backward", 9)])
                self.assertTrue(self.tool_results(reverse)[1].result.startswith("BLOCKED:"))
                self.assertEqual(self.tool_results(independent)[0].result, "moved backward for 9 meters.")

    async def test_safe_simulated_motor_tool_reuse_and_owned_sdk_lifecycle(self):
        agent = examples.build_agent(self.sdk, "AgentFunctionCallingMiddleware", emit=self.events.append)
        self.assertEqual({function.name for function in agent.default_options["tools"]}, {"forward", "backward", "stop", "turn_left", "turn_right"})
        self.payloads = [completion("", [("forward", {"distance": 3})]), completion()]
        with patch("shared.motor_tools._DELAY_SECONDS", 0):
            response = await agent.run("Move")
        self.assertEqual(self.tool_results(response)[0].result, "moved forward for 3 meters.")
        with patch.object(examples, "create_sdk_client", return_value=self.sdk):
            await examples.run_demos("AgentWithLogging")
        self.assertTrue(self.sdk.is_closed())


if __name__ == "__main__":
    unittest.main()
