"""Offline chapter 12 integration tests using real framework graphs and agents."""

import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch
import warnings

import httpx
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with warnings.catch_warnings(), patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
    warnings.simplefilter("ignore")
    from shared import sequential_concurrent_examples as examples
from agent_framework import (
    Agent, AgentExecutorResponse, AgentResponse, BaseChatClient, ChatResponse, ChatResponseUpdate,
    Content, FileCheckpointStorage, FunctionInvocationContext, Message, ResponseStream,
    Workflow, WorkflowBuilder, WorkflowErrorDetails, WorkflowEvent, WorkflowExecutor, tool,
)
from agent_framework.exceptions import WorkflowCheckpointException
from agent_framework import observability

ENV_REPORT = "Mission: avoid the tree and return to path. Temperature 20; humidity 40; droplets Low; wind 10."
ROUTE = "turn left 90\nforward 2\nturn right 90\nforward 4\nstop"
MOVES = "Executed: left 90, forward 2, right 90, forward 4, stop."
NAMES = ("EnvironmentAgent", "SafetyAgent", "MotorsAgent")


def clearance(state="GRANTED", reason="safe"):
    return json.dumps({"clearance": state, "reason": reason})


def agent_name(instructions):
    if "MaintenanceAgent" in instructions:
        return "MaintenanceAgent"
    if "EnvironmentAgent" in instructions:
        return "EnvironmentAgent"
    if "NavigatorAgent" in instructions:
        return "NavigatorAgent"
    if "SafetyAgent" in instructions:
        return "SafetyAgent"
    if "MotorsAgent" in instructions:
        return "MotorsAgent"
    raise AssertionError(f"Unrecognized source agent: {instructions}")


class ScriptedClient(BaseChatClient):
    def __init__(self, outputs=None, *, barrier=False):
        super().__init__()
        self.outputs = outputs or {}
        self.calls = []
        self.completed = []
        self.arrived = set()
        self.barrier = asyncio.Event() if barrier else None
        self.active = 0
        self.max_active = 0

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        name = agent_name(options.get("instructions", ""))
        self.calls.append((name, list(messages), dict(options)))

        async def respond():
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if self.barrier is not None:
                    self.arrived.add(name)
                    if self.arrived == {"MaintenanceAgent", "EnvironmentAgent"}:
                        self.barrier.set()
                    await asyncio.wait_for(self.barrier.wait(), 2)
                value = self.outputs.get(name)
                if value is None:
                    if "SAFETY THRESHOLDS" in options.get("instructions", ""):
                        value = clearance()
                    else:
                        value = {"EnvironmentAgent": ENV_REPORT, "SafetyAgent": "GRANTED: no rain",
                                 "NavigatorAgent": ROUTE, "MotorsAgent": MOVES}[name]
                if isinstance(value, Exception):
                    raise value
                self.completed.append(name)
                if isinstance(value, list):
                    return ChatResponse(messages=value)
                return ChatResponse(messages=[Message("assistant", [value], author_name=name)])
            finally:
                self.active -= 1

        if not stream:
            return respond()

        async def updates():
            response = await respond()
            for index, message in enumerate(response.messages):
                for content in message.contents:
                    if content.type == "text":
                        middle = len(content.text) // 2
                        for text in (content.text[:middle], content.text[middle:]):
                            yield ChatResponseUpdate(role=message.role, author_name=name,
                                                     response_id=f"r-{name}", message_id=f"m-{name}-{index}",
                                                     contents=[Content.from_text(text)])
                    else:
                        yield ChatResponseUpdate(role=message.role, author_name=name,
                                                 response_id=f"r-{name}", message_id=f"m-{name}-{index}",
                                                 contents=[content])

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)


def completion(text, calls=()):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [{"id": f"call-{name}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(arguments)}}
                                 for name, arguments in calls]
    return {"id": "chatcmpl-offline", "object": "chat.completion", "created": 1700000000, "model": "offline",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}


def sse(payload):
    message = payload["choices"][0]["message"]
    deltas = [{"role": "assistant"}]
    if message.get("tool_calls"):
        deltas.append({"tool_calls": [{"index": index, **call} for index, call in enumerate(message["tool_calls"])]})
    if message.get("content"):
        text = message["content"]
        middle = len(text) // 2
        deltas.extend(({"content": text[:middle]}, {"content": text[middle:]}))
    base = {name: payload[name] for name in ("id", "created", "model")}
    rows = [{**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            for delta in deltas]
    rows.append({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {},
                 "finish_reason": payload["choices"][0]["finish_reason"]}]})
    return ("".join(f"data: {json.dumps(row)}\n\n" for row in rows) + "data: [DONE]\n\n").encode()


def branch(name, messages):
    return AgentExecutorResponse(name, AgentResponse(messages=messages), [Message("user", ["mission"]), *messages])


def outputs(events):
    return [event.data for event in events if event.type == "output"]


def texts(messages):
    return [message.text for message in messages]


class SequentialConcurrentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_CHAT_MODEL_ID": "offline"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        telemetry = patch.object(observability, "OBSERVABILITY_SETTINGS",
                                 observability.ObservabilitySettings(enable_instrumentation=False))
        telemetry.start()
        self.addCleanup(telemetry.stop)
        for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError(f"Offline test attempted {target}"))
            guard.start()
            self.addCleanup(guard.stop)
        self.console = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.console)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)
        previous_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous_logging)
        self.requests = []
        self.executed = []

    def build(self, sample=examples.SAMPLES[0], client=None, **kwargs):
        return examples.build_sample(sample, client or ScriptedClient(), sensor_tools=[], motor_tools=[],
                                     maintenance_tools=[], **kwargs)

    async def collect(self, workflow, *, stream=False, prompt=examples.MISSION_PROMPT):
        if stream:
            return [event async for event in workflow.run(prompt, stream=True)]
        return await workflow.run(prompt)

    async def http_client(self, handler):
        async def request(request):
            body = json.loads(request.content)
            self.requests.append(body)
            payload = handler(body)
            if body.get("stream"):
                return httpx.Response(200, content=sse(payload), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=payload)
        transport = httpx.AsyncClient(transport=httpx.MockTransport(request))
        sdk = AsyncOpenAI(api_key="offline-not-a-secret", base_url="https://offline.invalid/v1", http_client=transport,
                          max_retries=0)
        self.addAsyncCleanup(sdk.close)
        return examples.create_completion_client(sdk)

    def tools(self):
        @tool(name="ReadDropletLevelAsync")
        async def read_droplets() -> str:
            """Read simulated raindrops."""
            self.executed.append("sensor")
            return "High"

        @tool(name="CheckBatteryHealthAsync")
        async def battery() -> str:
            """Read simulated battery health."""
            self.executed.append("maintenance")
            return "Battery health is at 60%."

        @tool(name="stop")
        async def stop() -> str:
            """Stop simulated motors."""
            self.executed.append("stop")
            return "stopped."
        return [read_droplets], [battery], [stop]

    async def test_all_eight_exact_named_source_graphs_execute(self):
        expected = ("SequentialOrchestration", "ProcessingPipeline", "RobbyMission", "SafeExecutionWithCheckpoints",
                    "SequentialOrchestrationWithIntermediateOutput", "SafeExecution", "SafetyAssessment",
                    "SanitizedSequentialExecution")
        for sample, name in zip(examples.SAMPLES, expected):
            with self.subTest(sample=sample):
                workflow = self.build(sample)
                self.assertIsInstance(workflow, Workflow)
                self.assertEqual(workflow.name, name)
                self.assertEqual(workflow.max_iterations, 20)
                result = await self.collect(workflow)
                self.assertEqual(len(outputs(result)), 1)
                self.assertIsInstance(outputs(result)[0], AgentResponse)
                self.assertIn("GRANTED" if sample == "ConcurrentOrchestration" else MOVES, outputs(result)[0].text)
                self.assertFalse(any(event.type in {"failed", "executor_failed"} for event in result))

    async def test_shared_history_is_chronological_and_final_output_is_full(self):
        client = ScriptedClient()
        events = await self.collect(self.build(client=client))
        self.assertEqual([call[0] for call in client.calls], list(NAMES))
        self.assertEqual(texts(client.calls[0][1]), [examples.MISSION_PROMPT])
        self.assertEqual(texts(client.calls[1][1]), [examples.MISSION_PROMPT, ENV_REPORT])
        self.assertEqual(texts(client.calls[2][1]), [examples.MISSION_PROMPT, ENV_REPORT, "GRANTED: no rain"])
        self.assertEqual(texts(outputs(events)[0].messages), [examples.MISSION_PROMPT, ENV_REPORT, "GRANTED: no rain", MOVES])
        self.assertEqual([message.role for message in outputs(events)[0].messages], ["user", "assistant", "assistant", "assistant"])

    async def test_pipeline_only_previous_response_and_last_agent_output(self):
        client = ScriptedClient()
        events = await self.collect(self.build(examples.SAMPLES[1], client))
        self.assertEqual([call[0] for call in client.calls], ["EnvironmentAgent", "NavigatorAgent", "MotorsAgent"])
        self.assertEqual(texts(client.calls[1][1]), [ENV_REPORT])
        self.assertEqual(texts(client.calls[2][1]), [ROUTE])
        self.assertEqual(texts(outputs(events)[0].messages), [MOVES])
        self.assertIn("including the original command", client.calls[0][2]["instructions"])
        self.assertIn("Do not reinterpret", client.calls[2][2]["instructions"])
        self.assertTrue(client.calls[2][2]["instructions"].startswith(examples.PIPELINE_MOTORS_INSTRUCTIONS))
        self.assertIn(examples.PIPELINE_TOOL_ADAPTATION, client.calls[2][2]["instructions"])
        self.assertEqual(client.calls[2][2]["tool_choice"], "required")
        self.assertFalse(client.calls[2][2]["allow_multiple_tool_calls"])
        self.assertEqual(client.calls[1][2]["tool_choice"], "auto")

    async def test_intermediate_buffered_events_are_selected_once(self):
        events = await self.collect(self.build(examples.SAMPLES[4]))
        data_events = [event for event in events if event.type in {"output", "intermediate"}]
        self.assertEqual([(event.type, event.executor_id) for event in data_events],
                         [("intermediate", "EnvironmentAgent"), ("intermediate", "SafetyAgent"), ("output", "MotorsAgent")])
        self.assertEqual([event.data.text for event in data_events], [ENV_REPORT, "GRANTED: no rain", MOVES])
        self.assertTrue(all(isinstance(event.data, AgentResponse) for event in data_events))

    async def test_intermediate_stream_routes_chunks_without_duplicate_final_output(self):
        events = await self.collect(self.build(examples.SAMPLES[2]), stream=True)
        data = [event for event in events if event.type in {"output", "intermediate"}]
        grouped = {}
        for event in data:
            grouped.setdefault((event.type, event.executor_id), []).append(event.data.text)
            self.assertIsInstance(event.data, examples.AgentResponseUpdate)
            examples.print_event(event)
        self.assertEqual({key: "".join(value) for key, value in grouped.items()},
                         {("intermediate", "EnvironmentAgent"): ENV_REPORT,
                          ("intermediate", "SafetyAgent"): "GRANTED: no rain", ("output", "MotorsAgent"): MOVES})
        self.assertEqual(len(data), 6)
        self.assertEqual(self.console.getvalue().count("[WORKFLOW OUTPUT]"), 2)
        self.assertEqual(self.console.getvalue().count("[INTERMEDIATE OUTPUT]"), 4)

    async def test_streamed_full_conversation_retains_order_and_output_type(self):
        client = ScriptedClient()
        events = await self.collect(self.build(client=client), stream=True)
        self.assertEqual(texts(outputs(events)[0].messages), [examples.MISSION_PROMPT, ENV_REPORT, "GRANTED: no rain", MOVES])
        self.assertEqual(len(outputs(events)), 1)
        self.assertFalse(any(event.type == "intermediate" for event in events))

    async def test_concurrent_branches_overlap_and_fanin_order_is_deterministic(self):
        client = ScriptedClient({"MaintenanceAgent": clearance("DENIED", "battery"),
                                 "EnvironmentAgent": clearance("DENIED", "wind")}, barrier=True)
        events = await self.collect(self.build(examples.SAMPLES[6], client))
        self.assertEqual(client.max_active, 2)
        self.assertEqual(client.arrived, {"MaintenanceAgent", "EnvironmentAgent"})
        self.assertEqual(outputs(events)[0].text, "DENIED: battery wind")
        self.assertTrue(all(texts(call[1]) == [examples.MISSION_PROMPT] for call in client.calls))
        self.assertEqual(len(outputs(events)), 1)

    async def test_concurrent_source_schema_thresholds_and_tool_metadata(self):
        client = ScriptedClient()
        workflow = examples.build_safety_workflow(client)
        events = await self.collect(workflow)
        self.assertEqual(outputs(events)[0].text, "GRANTED: All clearances were granted.")
        self.assertEqual([call[2]["response_format"] for call in client.calls], [examples.ClearanceResponse] * 2)
        maintenance = workflow.executors["MaintenanceAgent"].agent
        environment = workflow.executors["EnvironmentAgent"].agent
        self.assertIn("below 30%", maintenance.default_options["instructions"])
        self.assertIn("above 36 PSI", maintenance.default_options["instructions"])
        self.assertIn("Droplet level is Extreme", environment.default_options["instructions"])
        self.assertEqual({function.name for function in maintenance.default_options["tools"]},
                         {"UpdateFirmwareAsync", "CheckBatteryHealthAsync", "CleanSolarPanelsAsync", "CheckTirePressureAsync",
                          "CheckMotorsAsync", "CalibrateSensorsAsync"})
        self.assertEqual({function.name for function in environment.default_options["tools"]},
                         {"ReadDropletLevelAsync", "ReadHumidityAsync", "ReadInfraredRadiationAsync", "ReadTemperatureAsync",
                          "ReadWindDirectionAsync", "ReadWindSpeedAsync"})

    def test_worst_clearance_score_and_denial_reason_aggregation(self):
        for first, second, expected in (
            ("GRANTED", "GRANTED", "GRANTED: All clearances were granted."),
            ("DENIED", "GRANTED", "DENIED: battery"),
            ("GRANTED", "DENIED", "DENIED: rain"),
            ("DENIED", "DENIED", "DENIED: battery rain"),
        ):
            with self.subTest(first=first, second=second):
                response = examples.aggregate_clearances([
                    branch("MaintenanceAgent", [Message("assistant", [clearance(first, "battery")])]),
                    branch("EnvironmentAgent", [Message("assistant", [clearance(second, "rain")])]),
                ])
                self.assertEqual(response.text, expected)
                self.assertEqual(response.messages[0].author_name, "SafetyAggregator")
        self.assertGreater(examples.CLEARANCE_SCORE[examples.ClearanceState.DENIED],
                           examples.CLEARANCE_SCORE[examples.ClearanceState.GRANTED])

    def test_aggregation_excludes_tool_pollution_but_keeps_all_assistant_reports(self):
        call = Content.from_function_call("call", "sensor", arguments={})
        result = Content.from_function_result("call", result="not JSON")
        messages = [Message("user", ["not JSON"]), Message("assistant", ["  "]),
                    Message("assistant", [call, "not JSON"]), Message("tool", [result]),
                    Message("assistant", [clearance("GRANTED", "first")]),
                    Message("assistant", [clearance("DENIED", "later")])]
        response = examples.aggregate_clearances([
            branch("MaintenanceAgent", messages),
            branch("EnvironmentAgent", [Message("assistant", [clearance()])]),
        ])
        self.assertEqual(response.text, "DENIED: later")
        self.assertEqual(len(messages[2].contents), 2)
        self.assertEqual(messages[3].contents[0].result, "not JSON")

    def test_invalid_safety_json_or_schema_never_grants_clearance(self):
        for value in ("not JSON", "null", "[]", '{"clearance":"MAYBE","reason":"x"}',
                      '{"clearance":"GRANTED"}', '{"clearance":"GRANTED","reason":42}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                examples.aggregate_clearances([
                    branch("MaintenanceAgent", [Message("assistant", [value])]),
                    branch("EnvironmentAgent", [Message("assistant", [clearance()])]),
                ])

    def test_incomplete_or_duplicate_safety_branches_fail_closed(self):
        valid = branch("MaintenanceAgent", [Message("assistant", [clearance()])])
        other = branch("EnvironmentAgent", [Message("assistant", [clearance()])])
        for results in ([], [valid], [valid, valid], [valid, other, other],
                        [valid, branch("EnvironmentAgent", [])],
                        [valid, branch("EnvironmentAgent", [Message("tool", ["not an answer"])])]):
            with self.subTest(branches=[result.executor_id for result in results]), self.assertRaises(ValueError):
                examples.aggregate_clearances(results)

    async def test_concurrent_provider_failure_does_not_emit_a_false_summary(self):
        client = ScriptedClient({"MaintenanceAgent": RuntimeError("maintenance unavailable")})
        with self.assertRaisesRegex(RuntimeError, "maintenance unavailable"):
            await self.collect(self.build(examples.SAMPLES[6], client))
        self.assertNotIn("GRANTED: All clearances", self.console.getvalue())
        self.assertNotIn("MotorsAgent", client.completed)

    async def test_nested_workflow_agent_preserves_original_mission_and_verdict(self):
        client = ScriptedClient({"EnvironmentAgent": clearance("DENIED", "rain")})
        workflow = self.build(examples.SAMPLES[7], client)
        events = await self.collect(workflow)
        motors_call = next(call for call in client.calls if call[0] == "MotorsAgent")
        self.assertEqual(texts(motors_call[1]), [examples.MISSION_PROMPT, "DENIED: rain"])
        self.assertEqual(texts(outputs(events)[0].messages), [examples.MISSION_PROMPT, "DENIED: rain", MOVES])
        self.assertEqual(workflow.executors["SafetyStage"].agent.name, "SafetyStage")
        self.assertIn("Call MaintenanceTools", client.calls[0][2]["instructions"])
        self.assertIn("Call SensorTools", client.calls[1][2]["instructions"])

    async def test_real_workflow_executor_wrapper_forwards_typed_summary(self):
        inner = self.build(examples.SAMPLES[6])
        wrapper = WorkflowExecutor(inner, id="SafetyWrapper", allow_direct_output=True)
        workflow = WorkflowBuilder(start_executor=wrapper, output_from=[wrapper], name="NestedSafety").build()
        events = await self.collect(workflow)
        self.assertEqual(len(outputs(events)), 1)
        self.assertIsInstance(outputs(events)[0], AgentResponse)
        self.assertEqual(outputs(events)[0].text, "GRANTED: All clearances were granted.")
        self.assertTrue(any(event.executor_id == "SafetyWrapper" and event.type == "executor_completed" for event in events))

    async def test_mixed_http_tools_denial_stops_motors_and_boundary_is_clean(self):
        sensor, maintenance, motor = self.tools()
        counts = {}

        def handler(body):
            instructions = "\n".join(message.get("content") or "" for message in body["messages"] if message["role"] == "system")
            name = agent_name(instructions)
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                function = {"EnvironmentAgent": "ReadDropletLevelAsync", "MaintenanceAgent": "CheckBatteryHealthAsync",
                            "MotorsAgent": "stop"}[name]
                if name == "MotorsAgent":
                    self.assertEqual([message["role"] for message in body["messages"]], ["system", "user", "assistant"])
                    self.assertEqual(body["messages"][-1]["content"], "DENIED: rain")
                    self.assertFalse(any("tool_calls" in message for message in body["messages"]))
                return completion(None, [(function, {})])
            self.assertEqual(body["messages"][-1]["role"], "tool")
            return completion(clearance("DENIED", "rain") if name == "EnvironmentAgent" else
                              (clearance() if name == "MaintenanceAgent" else "Mission stopped due to unsafe conditions."))

        client = await self.http_client(handler)
        workflow = examples.build_sample(examples.SAMPLES[7], client, sensor_tools=sensor,
                                         maintenance_tools=maintenance, motor_tools=motor)
        events = await self.collect(workflow)
        self.assertEqual(set(self.executed), {"sensor", "maintenance", "stop"})
        self.assertEqual(self.executed[-1], "stop")
        self.assertEqual(counts, {"MaintenanceAgent": 2, "EnvironmentAgent": 2, "MotorsAgent": 2})
        self.assertEqual(texts(outputs(events)[0].messages)[-1], "Mission stopped due to unsafe conditions.")
        self.assertIn("DENIED: rain", texts(outputs(events)[0].messages))

    async def test_mixed_granted_path_executes_ordered_moves_after_both_safety_branches(self):
        sensor, maintenance, _ = self.tools()

        @tool(name="turn_left")
        async def turn_left(angle: int) -> str:
            """Turn simulated motors anticlockwise."""
            self.executed.append(("turn_left", angle))
            return f"turned anticlockwise {angle} degrees."

        @tool(name="forward")
        async def forward(distance: int) -> str:
            """Move simulated motors forward."""
            self.executed.append(("forward", distance))
            return f"moved forward for {distance} meters."

        counts = {}
        route = [("turn_left", {"angle": 90}), ("forward", {"distance": 2})]

        def handler(body):
            name = agent_name(body["messages"][0]["content"])
            counts[name] = counts.get(name, 0) + 1
            if name == "MotorsAgent":
                self.assertEqual(counts["MaintenanceAgent"], 2)
                self.assertEqual(counts["EnvironmentAgent"], 2)
                if counts[name] == 1:
                    self.assertEqual(body["messages"][-1]["content"], "GRANTED: All clearances were granted.")
                if counts[name] <= len(route):
                    return completion(None, [route[counts[name] - 1]])
                return completion("Executed: turn left 90, forward 2.")
            if counts[name] == 1:
                return completion(None, [("ReadDropletLevelAsync" if name == "EnvironmentAgent" else "CheckBatteryHealthAsync", {})])
            return completion(clearance())

        client = await self.http_client(handler)
        events = await self.collect(examples.build_sample(examples.SAMPLES[7], client, sensor_tools=sensor,
                                                         maintenance_tools=maintenance, motor_tools=[turn_left, forward]))
        self.assertEqual(self.executed[-2:], [("turn_left", 90), ("forward", 2)])
        self.assertEqual(set(self.executed[:2]), {"sensor", "maintenance"})
        self.assertEqual(outputs(events)[0].messages[-1].text, "Executed: turn left 90, forward 2.")
        self.assertEqual(counts["MotorsAgent"], 3)

    async def test_http_shared_history_retains_paired_tool_results_and_executes_motor(self):
        sensor, _, motor = self.tools()
        counts = {}

        def handler(body):
            name = agent_name(body["messages"][0]["content"])
            counts[name] = counts.get(name, 0) + 1
            if name == "EnvironmentAgent" and counts[name] == 1:
                return completion(None, [("ReadDropletLevelAsync", {})])
            if name == "MotorsAgent" and counts[name] == 1:
                self.assertEqual(body["messages"][1]["content"], examples.MISSION_PROMPT)
                self.assertTrue(any(message["role"] == "tool" and message["content"] == "High" for message in body["messages"]))
                self.assertIn("DENIED: rain", [message.get("content") for message in body["messages"]])
                return completion(None, [("stop", {})])
            return completion({"EnvironmentAgent": "Droplets High", "SafetyAgent": "DENIED: rain", "MotorsAgent": "stopped"}[name])

        client = await self.http_client(handler)
        workflow = examples.build_sample(examples.SAMPLES[0], client, sensor_tools=sensor, motor_tools=motor)
        events = await self.collect(workflow)
        self.assertEqual(self.executed, ["sensor", "stop"])
        messages = outputs(events)[0].messages
        self.assertEqual(messages[0].text, examples.MISSION_PROMPT)
        self.assertEqual(messages[-1].text, "stopped")
        self.assertEqual([content.result for message in messages for content in message.contents if content.type == "function_result"],
                         ["High", "stopped."])

    async def test_http_streaming_pipeline_only_last_response_and_tool_result_loop(self):
        sensor, _, motor = self.tools()
        counts = {}

        def handler(body):
            self.assertTrue(body["stream"])
            name = agent_name(body["messages"][0]["content"])
            counts[name] = counts.get(name, 0) + 1
            if name == "EnvironmentAgent" and counts[name] == 1:
                return completion(None, [("ReadDropletLevelAsync", {})])
            if name == "NavigatorAgent":
                self.assertNotIn(examples.MISSION_PROMPT, [message.get("content") for message in body["messages"]])
                return completion(ROUTE)
            if name == "MotorsAgent":
                self.assertFalse(body["parallel_tool_calls"])
                self.assertTrue(body["messages"][0]["content"].startswith(examples.PIPELINE_MOTORS_INSTRUCTIONS))
                self.assertIn(examples.PIPELINE_TOOL_ADAPTATION, body["messages"][0]["content"])
                if counts[name] == 1:
                    self.assertEqual(body["tool_choice"], "required")
                    self.assertEqual([message.get("content") for message in body["messages"][1:]], [ROUTE])
                    return completion(None, [("stop", {})])
                self.assertIn(body.get("tool_choice"), (None, "auto"))
            return completion(ENV_REPORT if name == "EnvironmentAgent" else "stopped")

        client = await self.http_client(handler)
        workflow = examples.build_sample(examples.SAMPLES[1], client, sensor_tools=sensor, motor_tools=motor)
        events = await self.collect(workflow, stream=True)
        final = outputs(events)[0].messages
        self.assertEqual([message.text for message in final if message.text], ["stopped"])
        motor_followup = self.requests[-1]["messages"]
        self.assertEqual(motor_followup[-1]["role"], "tool")
        self.assertEqual(motor_followup[-1]["content"], "stopped.")
        self.assertFalse(any(message.text == examples.MISSION_PROMPT or message.text == ENV_REPORT for message in final))
        self.assertEqual(self.executed, ["sensor", "stop"])
        self.assertEqual(len(outputs(events)), 1)

    async def test_pipeline_rejects_prose_only_completion_and_cli_exits_nonzero(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                def handler(body):
                    name = agent_name(body["messages"][0]["content"])
                    return completion({"EnvironmentAgent": ENV_REPORT, "NavigatorAgent": ROUTE, "MotorsAgent": MOVES}[name])

                client = await self.http_client(handler)
                _, _, motor = self.tools()
                workflow = examples.build_sample(examples.SAMPLES[1], client, sensor_tools=[], motor_tools=motor)
                with self.assertRaisesRegex(RuntimeError, "Incomplete pipeline motor execution"):
                    await self.collect(workflow, stream=stream)
                self.assertEqual(self.executed, [])
                self.assertNotIn("[WORKFLOW OUTPUT]", self.console.getvalue())

        def incomplete(coroutine):
            coroutine.close()
            raise RuntimeError("Incomplete pipeline motor execution: prose is not execution.")

        with patch.object(examples.asyncio, "run", side_effect=incomplete):
            code = examples.main(examples.SAMPLES[1], sample_dir=ROOT, argv=[])
        self.assertEqual(code, 1)
        self.assertIn("prose is not execution", self.console.getvalue())

    async def test_pipeline_real_motor_tools_execute_in_order_in_both_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.executed.clear()
                counts = {}
                plan = "turn left 90\nforward 2\nstop"

                @tool(name="turn_left")
                async def turn_left(angle: int) -> str:
                    """Turn simulated motors anticlockwise."""
                    self.executed.append(("turn_left", angle))
                    return f"turned anticlockwise {angle} degrees."

                @tool(name="forward")
                async def forward(distance: int) -> str:
                    """Move simulated motors forward."""
                    self.executed.append(("forward", distance))
                    return f"moved forward for {distance} meters."

                _, _, stop_tools = self.tools()
                movement_tools = [turn_left, forward, *stop_tools]
                route = [("turn_left", {"angle": 90}), ("forward", {"distance": 2}), ("stop", {})]

                def handler(body):
                    name = agent_name(body["messages"][0]["content"])
                    counts[name] = counts.get(name, 0) + 1
                    if name != "MotorsAgent":
                        return completion(ENV_REPORT if name == "EnvironmentAgent" else plan)
                    self.assertFalse(body["parallel_tool_calls"])
                    if counts[name] == 1:
                        self.assertEqual(body["tool_choice"], "required")
                        self.assertEqual([message.get("content") for message in body["messages"][1:]], [plan])
                    else:
                        self.assertEqual(body["messages"][-1]["role"], "tool")
                        self.assertIn(body.get("tool_choice"), (None, "auto"))
                    if counts[name] <= len(route):
                        return completion(None, [route[counts[name] - 1]])
                    return completion("Executed: turn left 90, forward 2, stop.")

                client = await self.http_client(handler)
                workflow = examples.build_sample(examples.SAMPLES[1], client, sensor_tools=[], motor_tools=movement_tools)
                events = await self.collect(workflow, stream=stream)
                self.assertEqual(self.executed, [("turn_left", 90), ("forward", 2), "stop"])
                self.assertEqual(counts["MotorsAgent"], 4)
                final = outputs(events)[0].messages
                self.assertEqual(final[-1].text, "Executed: turn left 90, forward 2, stop.")
                self.assertFalse(any(message.text in (examples.MISSION_PROMPT, ENV_REPORT, plan) for message in final))

    async def test_pipeline_failed_motor_attempt_does_not_claim_success(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                counts = {}
                attempts = []

                @tool(name="forward")
                async def failed_forward(distance: int) -> str:
                    """Move simulated motors forward."""
                    attempts.append(distance)
                    raise RuntimeError("motor unavailable")

                _, _, stop_tools = self.tools()

                def handler(body):
                    name = agent_name(body["messages"][0]["content"])
                    counts[name] = counts.get(name, 0) + 1
                    if name != "MotorsAgent":
                        return completion(ENV_REPORT if name == "EnvironmentAgent" else "forward 2\nstop")
                    if counts[name] == 1:
                        return completion(None, [("forward", {"distance": 2})])
                    if counts[name] == 2:
                        return completion(None, [("stop", {})])
                    return completion("Executed: forward 2, stop.")

                client = await self.http_client(handler)
                workflow = examples.build_sample(examples.SAMPLES[1], client, sensor_tools=[],
                                                 motor_tools=[failed_forward, *stop_tools])
                with self.assertRaisesRegex(RuntimeError, "1 successful motor calls out of 2 attempted"):
                    await self.collect(workflow, stream=stream)
                self.assertEqual(attempts, [2])

    async def test_exact_whole_message_filter_preserves_safe_content_without_mutation(self):
        safe = Message("user", ["mission", Content.from_text("detail")])
        mixed = Message("assistant", ["must drop this text", Content.from_function_call("call", "sensor", arguments={})])
        result = Message("tool", [Content.from_function_result("call", result="must drop")])
        verdict = Message("assistant", ["DENIED: rain"])
        source = [safe, mixed, result, verdict]
        clean = examples.without_tool_call_messages(source)
        self.assertEqual(clean, [safe, verdict])
        self.assertIs(clean[0], safe)
        self.assertEqual(len(mixed.contents), 2)
        self.assertEqual(len(source), 4)
        client = ScriptedClient()
        agent = Agent(client=client, name="MotorsAgent", instructions=examples.MOTORS_INSTRUCTIONS,
                      middleware=[examples.ToolCallFiltering()])
        for stream in (False, True):
            with self.subTest(stream=stream):
                if stream:
                    await agent.run(source, stream=True).get_final_response()
                else:
                    await agent.run(source)
                self.assertEqual(texts(client.calls[-1][1]), ["mission detail", "DENIED: rain"])
                self.assertFalse(any(content.type in {"function_call", "function_result"}
                                     for message in client.calls[-1][1] for content in message.contents))

    async def test_middleware_granted_pre_post_order_for_both_execution_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                trace = []
                client = examples.MiddlewareChatClient(ScriptedClient(), middleware=[examples.MissionAbortChat(trace=trace)])
                agent = Agent(client=client, name="SafetyAgent", instructions=examples.SAFETY_INSTRUCTIONS,
                              middleware=[examples.MissionAbortAgent(trace=trace)])
                response = await agent.run("mission", stream=True).get_final_response() if stream else await agent.run("mission")
                self.assertEqual(response.text, "GRANTED: no rain")
                self.assertEqual(trace, ["agent:pre", "chat:pre", "chat:post", "agent:post"])

    async def test_middleware_denial_aborts_graph_before_motor_for_both_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                client = ScriptedClient({"SafetyAgent": "DENIED: rain"})
                with self.assertRaisesRegex(examples.MissionAbort, "MISSION ABORT"):
                    await self.collect(self.build(examples.SAMPLES[5], client), stream=stream)
                self.assertEqual([call[0] for call in client.calls], ["EnvironmentAgent", "SafetyAgent"])
                self.assertIn("*** MISSION ABORT ***", self.console.getvalue())

    async def test_agent_abort_structured_invalid_json_and_case_sensitive_plain_checks(self):
        for text, structured, error in (("DENIED: rain", False, "MISSION ABORT"),
                                        (clearance("DENIED", "rain"), True, "MISSION ABORT"),
                                        ("bad JSON", True, "invalid JSON")):
            with self.subTest(text=text):
                agent = Agent(client=ScriptedClient({"SafetyAgent": text}), name="SafetyAgent",
                              instructions=examples.SAFETY_INSTRUCTIONS,
                              middleware=[examples.MissionAbortAgent(structured=structured)])
                with self.assertRaisesRegex(examples.MissionAbort, error):
                    await agent.run("mission")
        response = examples._check_clearance(AgentResponse(messages=[Message("assistant", ["denied lowercase"])]))
        self.assertEqual(response.text, "denied lowercase")

    async def test_source_failure_helpers_skip_inner_calls_and_report_matching_layer(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                inner = ScriptedClient()
                client = examples.MiddlewareChatClient(inner, middleware=[examples.SimulateAgentFailure()])
                agent = Agent(client=client, name="SafetyAgent", instructions=examples.SAFETY_INSTRUCTIONS)
                with self.assertRaisesRegex(RuntimeError, "chat response middleware"):
                    if stream:
                        await agent.run("mission", stream=True).get_final_response()
                    else:
                        await agent.run("mission")
                self.assertEqual(inner.calls, [])
        next_call = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "function calling middleware"):
            await examples.SimulateToolFailure().process(Mock(spec=FunctionInvocationContext), next_call)
        next_call.assert_not_awaited()

    async def test_checkpoint_default_completes_then_replays_only_motors_on_fresh_graph(self):
        client = ScriptedClient()
        built = []
        with tempfile.TemporaryDirectory() as directory:
            storage = FileCheckpointStorage(Path(directory), allowed_checkpoint_types=[])

            def build(**kwargs):
                workflow = self.build(examples.SAMPLES[3], client, **kwargs)
                built.append(workflow)
                return workflow

            result = await examples.run_checkpoint_demo(build, storage, emit=lambda _: None)
            self.assertEqual([call[0] for call in client.calls], [*NAMES, "MotorsAgent"])
            self.assertEqual(texts(client.calls[-1][1]), [examples.MISSION_PROMPT, ENV_REPORT, "GRANTED: no rain"])
            self.assertEqual([workflow.name for workflow in built], ["SafeExecutionWithCheckpoints", "SafeExecutionFromCheckpoint"])
            self.assertIsNot(built[0].executors["MotorsAgent"], built[1].executors["MotorsAgent"])
            checkpoint = await storage.load(result.checkpoint_id)
            self.assertIn("MotorsAgent", examples.pending_executor_ids(built[0], checkpoint))
            self.assertTrue(checkpoint.state)
            self.assertTrue(list(Path(directory).glob("*.json")))
            self.assertEqual(texts(outputs(result.resumed_events)[0].messages),
                             [examples.MISSION_PROMPT, ENV_REPORT, "GRANTED: no rain", MOVES])

    async def test_checkpoint_interruption_and_resume_with_new_storage_client_and_graph(self):
        first_client = ScriptedClient()
        resumed_client = ScriptedClient()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first_storage = FileCheckpointStorage(path, allowed_checkpoint_types=[])
            result = await examples.run_checkpoint_demo(
                lambda **kwargs: self.build(examples.SAMPLES[3], first_client, **kwargs), first_storage,
                interrupt=True, emit=lambda _: None)
            self.assertTrue(result.interrupted)
            self.assertEqual(result.resumed_events, [])
            self.assertEqual([call[0] for call in first_client.calls], ["EnvironmentAgent", "SafetyAgent"])
            self.assertNotEqual(result.checkpoint_id, "")
            storage = FileCheckpointStorage(path, allowed_checkpoint_types=[])
            resumed = await examples.run_checkpoint_demo(
                lambda **kwargs: self.build(examples.SAMPLES[3], resumed_client, **kwargs), storage,
                resume_id=result.checkpoint_id, emit=lambda _: None)
            self.assertEqual([call[0] for call in resumed_client.calls], ["MotorsAgent"])
            self.assertEqual(texts(resumed_client.calls[0][1]), [examples.MISSION_PROMPT, ENV_REPORT, "GRANTED: no rain"])
            self.assertEqual(texts(outputs(resumed.resumed_events)[0].messages)[-1], MOVES)
            self.assertEqual(resumed.first_events, [])

    async def test_checkpoint_http_sensor_tool_results_survive_fresh_graph_resume(self):
        sensor, _, motor = self.tools()
        counts = {}

        def handler(body):
            name = agent_name(body["messages"][0]["content"])
            counts[name] = counts.get(name, 0) + 1
            if name == "EnvironmentAgent" and counts[name] == 1:
                return completion(None, [("ReadDropletLevelAsync", {})])
            if name == "MotorsAgent" and counts[name] == 1:
                self.assertEqual(counts, {"EnvironmentAgent": 2, "SafetyAgent": 1, "MotorsAgent": 1})
                self.assertTrue(any(message.get("tool_call_id") == "call-ReadDropletLevelAsync" and
                                    message.get("content") == "High" for message in body["messages"]))
                self.assertEqual(body["messages"][-1]["content"], "DENIED: rain")
                return completion(None, [("stop", {})])
            return completion({"EnvironmentAgent": "Droplets High", "SafetyAgent": "DENIED: rain", "MotorsAgent": "stopped"}[name])

        client = await self.http_client(handler)
        with tempfile.TemporaryDirectory() as directory:
            storage = FileCheckpointStorage(Path(directory), allowed_checkpoint_types=[])

            def build(**kwargs):
                return examples.build_sample(examples.SAMPLES[3], client, sensor_tools=sensor, motor_tools=motor, **kwargs)

            first = await examples.run_checkpoint_demo(build, storage, interrupt=True, emit=lambda _: None)
            self.assertEqual(self.executed, ["sensor"])
            resumed = await examples.run_checkpoint_demo(build, FileCheckpointStorage(Path(directory), allowed_checkpoint_types=[]),
                                                        resume_id=first.checkpoint_id, emit=lambda _: None)
            self.assertEqual(self.executed, ["sensor", "stop"])
            final = outputs(resumed.resumed_events)[0].messages
            self.assertEqual(final[0].text, examples.MISSION_PROMPT)
            self.assertEqual(final[-1].text, "stopped")
            self.assertTrue(any(content.type == "function_result" and content.result == "High"
                                for message in final for content in message.contents))

    async def test_checkpoint_wrong_topology_and_unknown_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = FileCheckpointStorage(Path(directory), allowed_checkpoint_types=[])
            result = await examples.run_checkpoint_demo(lambda **kwargs: self.build(examples.SAMPLES[3], **kwargs),
                                                        storage, interrupt=True, emit=lambda _: None)
            with self.assertRaisesRegex(WorkflowCheckpointException, "graph has changed"):
                await self.build(examples.SAMPLES[1]).run(checkpoint_id=result.checkpoint_id, checkpoint_storage=storage)
            with self.assertRaisesRegex(WorkflowCheckpointException, "No checkpoint found"):
                await self.build(examples.SAMPLES[3]).run(checkpoint_id="missing", checkpoint_storage=storage)

    async def test_source_sensor_ranges_enum_values_tool_names_and_delay(self):
        rng = Mock()
        rng.randrange.side_effect = [-20, 99, 0, 99]
        rng.choice.side_effect = [examples.DropletLevel.HIGH, examples.Direction.SOUTH_EAST]
        sensors = examples.SensorTools(rng=rng)
        with patch.object(examples.asyncio, "sleep", new=AsyncMock()) as sleep:
            values = [await sensors.read_temperature(), await sensors.read_infrared_radiation(), await sensors.read_humidity(),
                      await sensors.read_droplet_level(), await sensors.read_wind_direction(), await sensors.read_wind_speed()]
        self.assertEqual(values, [-20, 99, 0, "High", "SouthEast", 99])
        self.assertEqual([call.args for call in rng.randrange.call_args_list], [(-20, 100), (0, 100), (0, 100), (0, 100)])
        self.assertEqual([call.args for call in sleep.await_args_list], [(0.5,)] * 6)
        self.assertEqual([value.value for value in examples.DropletLevel], ["None", "Low", "Medium", "High"])
        self.assertEqual(len(examples.Direction), 8)
        self.assertEqual([function.name for function in sensors.as_tools()],
                         ["ReadDropletLevelAsync", "ReadHumidityAsync", "ReadInfraredRadiationAsync", "ReadTemperatureAsync",
                          "ReadWindDirectionAsync", "ReadWindSpeedAsync"])

    async def test_source_maintenance_result_ranges_and_delay_multipliers(self):
        rng = Mock()
        rng.randrange.side_effect = [29, 34, 0]
        maintenance = examples.MaintenanceTools(rng=rng)
        with patch.object(examples.asyncio, "sleep", new=AsyncMock()) as sleep:
            values = [await maintenance.calibrate_sensors(), await maintenance.check_motors(),
                      await maintenance.check_tire_pressure(), await maintenance.clean_solar_panels(),
                      await maintenance.check_battery_health(), await maintenance.update_firmware()]
        self.assertEqual(values, ["All sensors have been calibrated.", "Motors checked. Status: 29% efficiency.",
                                 "Tire pressure is 34 PSI.", "Solar panels have been cleaned.",
                                 "Battery health is at 0%.", "Firmware has been updated to the latest version."])
        self.assertEqual([call.args for call in rng.randrange.call_args_list], [(0, 100), (30, 35), (0, 100)])
        self.assertEqual([call.args for call in sleep.await_args_list], [(0.5,), (0.5,), (0.5,), (1.5,), (0.5,), (2.5,)])
        self.assertEqual([function.name for function in maintenance.as_tools()],
                         ["UpdateFirmwareAsync", "CheckBatteryHealthAsync", "CleanSolarPanelsAsync", "CheckTirePressureAsync",
                          "CheckMotorsAsync", "CalibrateSensorsAsync"])

    def test_typed_output_and_event_field_routing(self):
        messages = [Message("user", ["mission"]), Message("tool", ["hide"]), Message("assistant", ["answer"])]
        examples.print_event(WorkflowEvent("output", data=AgentResponse(messages=messages), executor_id="MotorsAgent"), all_messages=True)
        examples.print_event(WorkflowEvent.superstep_completed(3))
        examples.print_event(WorkflowEvent.executor_completed("SafetyAgent"))
        examples.print_event(WorkflowEvent.executor_failed("SafetyAgent", WorkflowErrorDetails("RuntimeError", "abort")))
        self.assertIn("user: mission\nassistant: answer", self.console.getvalue())
        self.assertNotIn("hide", self.console.getvalue())
        self.assertIn("[SUPER STEP] 3", self.console.getvalue())
        self.assertIn("[EXECUTOR] SafetyAgent completed.", self.console.getvalue())
        self.assertIn("[WORKFLOW ERROR] SafetyAgent: abort", self.console.getvalue())
        for value in ("string is not a conversation", [messages[0], "wrong"], {"text": "wrong"}):
            with self.subTest(type=type(value).__name__), self.assertRaises(TypeError):
                examples.output_messages(value)
        self.assertEqual(examples.output_messages(messages), messages)
        self.assertEqual(examples.output_messages(messages[0]), [messages[0]])

    async def test_cli_provider_sdk_lifecycle_and_source_defaults(self):
        sdk = Mock()
        sdk.__aenter__ = AsyncMock(return_value=sdk)
        sdk.__aexit__ = AsyncMock(return_value=False)
        client = ScriptedClient()
        args = examples.argument_parser(examples.SAMPLES[0]).parse_args(["--provider", "azure"])
        workflow = self.build(client=client)
        settings = observability.ObservabilitySettings(enable_instrumentation=True, enable_sensitive_data=True,
                                                      enable_console_exporters=True, vs_code_extension_port=4317)
        with patch.object(examples, "create_sdk_client", return_value=sdk) as create_sdk, \
                patch.object(examples, "create_completion_client", return_value=client) as completion_client, \
                patch.object(examples, "build_sample", return_value=workflow), \
                patch.object(observability, "OBSERVABILITY_SETTINGS", settings), \
                patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://offline.invalid:4317",
                                        "ENABLE_INSTRUMENTATION": "true", "ENABLE_SENSITIVE_DATA": "true",
                                        "ENABLE_CONSOLE_EXPORTERS": "true"}), \
                patch.object(observability, "_create_otlp_exporters") as exporters:
            await examples.run_sample(examples.SAMPLES[0], args, sample_dir=ROOT / "12.SequentialAndConcurrent")
            self.assertTrue(settings.is_user_disabled)
            self.assertFalse(settings.enable_instrumentation)
            self.assertFalse(settings.enable_sensitive_data)
            exporters.assert_not_called()
        create_sdk.assert_called_once_with("azure")
        completion_client.assert_called_once_with(sdk, "azure")
        sdk.__aexit__.assert_awaited_once()
        self.assertEqual(texts(client.calls[0][1]), [examples.MISSION_PROMPT])
        self.assertEqual(examples.argument_parser(examples.SAMPLES[6]).parse_args([]).prompt, examples.ASSESSMENT_PROMPT)
        self.assertEqual(examples.argument_parser(examples.SAMPLES[7]).parse_args([]).prompt, examples.MIXED_PROMPT)

    async def test_cli_bounded_timeout_validation_and_provider_errors_exit_cleanly(self):
        for timeout in ("0", "-1", "nan", "inf"):
            with self.subTest(timeout=timeout), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(examples, "create_sdk_client") as sdk, self.assertRaises(SystemExit) as exited:
                examples.main(examples.SAMPLES[0], sample_dir=ROOT, argv=["--timeout", timeout])
            self.assertEqual(exited.exception.code, 2)
            sdk.assert_not_called()

        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        args = examples.argument_parser(examples.SAMPLES[0]).parse_args(["--timeout", "0.01"])
        with patch.object(examples, "run_sample", side_effect=slow) as run, self.assertRaises(TimeoutError):
            await examples.run_bounded(examples.SAMPLES[0], args, sample_dir=ROOT)
        run.assert_called_once()
        with patch.object(examples, "create_sdk_client", side_effect=RuntimeError("OPENAI_API_KEY is not set")), \
                self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is not set"):
            await examples.run_bounded(examples.SAMPLES[0], args, sample_dir=ROOT)

        def fail(coroutine):
            coroutine.close()
            raise RuntimeError("OPENAI_API_KEY is not set")

        with patch.object(examples.asyncio, "run", side_effect=fail):
            code = examples.main(examples.SAMPLES[0], sample_dir=ROOT, argv=[])
        self.assertEqual(code, 1)
        self.assertIn("OPENAI_API_KEY is not set", self.console.getvalue())

    async def test_source_attaches_existing_motor_tools_only_to_motor_agents(self):
        for sample in (examples.SAMPLES[0], examples.SAMPLES[1], examples.SAMPLES[6], examples.SAMPLES[7]):
            with self.subTest(sample=sample):
                with patch.object(examples, "create_motor_tools", wraps=examples.create_motor_tools) as motor_factory:
                    workflow = examples.build_sample(sample, ScriptedClient())
                self.assertEqual(motor_factory.call_count, 0 if sample == "ConcurrentOrchestration" else 1)
                if sample != "ConcurrentOrchestration":
                    motors = workflow.executors["MotorsAgent"].agent
                    self.assertEqual([function.name for function in motors.default_options["tools"]],
                                     ["turn_right", "turn_left", "stop", "forward", "backward"])
                if "SafetyAgent" in workflow.executors:
                    self.assertEqual(workflow.executors["SafetyAgent"].agent.default_options.get("tools"), [])

    def test_eight_entrypoint_help_is_offline_and_checkpoint_runtime_is_ignored(self):
        for sample in examples.SAMPLES:
            with self.subTest(sample=sample):
                path = ROOT / "12.SequentialAndConcurrent" / sample / "main.py"
                self.assertTrue(path.is_file())
                spec = importlib.util.spec_from_file_location(f"chapter12_{sample}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with self.assertRaises(SystemExit) as exit_result:
                    module.main(sample, sample_dir=path.parent, argv=["--help"])
                self.assertEqual(exit_result.exception.code, 0)
                help_text = " ".join(examples.argument_parser(sample).format_help().split())
                self.assertIn("20 workflow supersteps", help_text)
                self.assertIn("120-second default timeout", help_text)
                self.assertIn("--provider", help_text)
        ignore = (ROOT / "12.SequentialAndConcurrent" / ".gitignore").read_text()
        self.assertIn("**/runtime/", ignore)
        parser = examples.argument_parser(examples.SAMPLES[3])
        self.assertEqual(parser.parse_args(["--resume", "id"]).resume, "id")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--resume", "id", "--interrupt"])


if __name__ == "__main__":
    unittest.main()
