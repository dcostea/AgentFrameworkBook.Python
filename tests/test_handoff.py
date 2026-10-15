"""Offline branch-routing and tool-evidence tests for chapter 14."""

import asyncio
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import AsyncOpenAI

with patch("dotenv.load_dotenv", return_value=False):
    from agent_framework import AgentResponse, Content, Message, WorkflowViz
    from shared import handoff_examples as handoff, motor_tools

ROOT = Path(__file__).resolve().parents[1]


def completion(text="Mission complete.", calls=None):
    return {
        "id": "handoff-test", "object": "chat.completion", "created": 1, "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": {
            "role": "assistant", "content": None if calls else text,
            **({"tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                               for cid, name, args in calls]} if calls else {}),
        }}],
    }


def stream_completion(text="Mission complete.", calls=None):
    base = {"id": "handoff-stream-test", "object": "chat.completion.chunk", "created": 1, "model": "test-model"}
    deltas = [{"role": "assistant"}]
    if calls:
        deltas.append({"tool_calls": [{"index": i, "id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                                     for i, (cid, name, args) in enumerate(calls)]})
    else:
        deltas.extend({"content": part} for part in (text[:5], text[5:]))
    chunks = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


class HandoffTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "OPENAI_CHAT_MODEL_ID": "test-model"}, clear=True))
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.object(motor_tools, "_DELAY_SECONDS", 0))
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request",
                                      new=AsyncMock(side_effect=AssertionError("Live network prohibited"))))

    def sdk(self, scenario="safe", *, stream=False, no_handoff=False):
        requests = []
        counters = {}
        actors = []
        selected = {"safe": "MotorsAgent", "fire": "FireDetectorAgent", "rain": "RainDetectorAgent"}[scenario]
        sensor = {"safe": (25, "None"), "fire": (85, "None"), "rain": (25, "High")}[scenario]

        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            instructions = body["messages"][0]["content"]
            if instructions.startswith("## PERSONA\nYou are the EnvironmentAgent"):
                actor = "EnvironmentAgent"
            elif instructions.startswith("You are the FireDetectorAgent"):
                actor = "FireDetectorAgent"
            elif instructions.startswith("You are the RainDetectorAgent"):
                actor = "RainDetectorAgent"
            else:
                actor = "MotorsAgent"
            actors.append(actor)
            turn = counters.get(actor, 0)
            counters[actor] = turn + 1
            calls = None
            if actor == "EnvironmentAgent":
                if no_handoff:
                    text = "Please specify the route."
                elif turn == 0:
                    calls = [("temperature", "read_temperature", {})]
                elif turn == 1:
                    calls = [("droplets", "read_droplet_level", {})]
                else:
                    calls = [("route", "handoff_to_"+selected, {})]
                    text = "Routing."
            elif actor == "FireDetectorAgent":
                if turn == 0:
                    calls = [("alarm", "sound_alarm", {})]
                elif turn == 1:
                    calls = [("sprinkle", "start_water_sprinkle", {})]
                else:
                    calls = [("fire-route", "handoff_to_MotorsAgent", {})]
                    text = "Fire handled."
            elif actor == "RainDetectorAgent":
                if turn == 0:
                    calls = [("wipers", "start_wipers", {"droplet_level": "High"})]
                else:
                    calls = [("rain-route", "handoff_to_MotorsAgent", {})]
                    text = "Rain handled."
            elif actor == "MotorsAgent":
                if turn == 0:
                    calls = [("motor", "forward", {"distance": 3})] if scenario == "safe" else [("motor", "stop", {})]
                else:
                    text = "Mission complete." if scenario == "safe" else "Hazard detected; stopped."
            if calls:
                return stream_completion(calls=calls) if stream else httpx.Response(200, json=completion(calls=calls))
            return stream_completion(text) if stream else httpx.Response(200, json=completion(text))

        sdk = AsyncOpenAI(api_key="offline", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        self.addAsyncCleanup(sdk.close)
        return sdk, requests, actors, handoff.SensorReadings(temperature=sensor[0], droplet_level=handoff.DropletLevel(sensor[1]))

    async def test_native_workflow_safe_fire_rain_routes_one_specialist(self):
        for scenario in ("safe", "fire", "rain"):
            for reasons in (False, True):
                with self.subTest(scenario=scenario, reasons=reasons):
                    sdk, requests, actors, readings = self.sdk(scenario)
                    workflow = handoff.build_workflow(sdk, reasons=reasons, readings=readings,
                                                      sensor_delay=0, fire_delay=0, rain_delay=0)
                    result = await workflow.run(handoff.PROMPT)
                    self.assertEqual(result.get_request_info_events(), [])
                    routes = [(e.data.source, e.data.target) for e in result if e.type == "handoff_sent"]
                    expected = [("EnvironmentAgent", "MotorsAgent")] if scenario == "safe" else [
                        ("EnvironmentAgent", "FireDetectorAgent" if scenario == "fire" else "RainDetectorAgent"),
                        ("FireDetectorAgent" if scenario == "fire" else "RainDetectorAgent", "MotorsAgent"),
                    ]
                    self.assertEqual(routes, expected)
                    self.assertNotIn("RainDetectorAgent" if scenario != "rain" else "FireDetectorAgent", actors)
                    self.assertEqual(actors[-1], "MotorsAgent")
                    outputs = result.get_outputs()
                    self.assertTrue(outputs)
                    self.assertTrue(any((isinstance(o, list) and any(m.text.endswith("Mission complete.") or m.text.endswith("Hazard detected; stopped.") for m in o))
                                        or getattr(o, "text", "").endswith(("Mission complete.", "Hazard detected; stopped.")) for o in outputs))
                    self.assertTrue(all(r["parallel_tool_calls"] is False for r in requests))

    async def test_native_handoff_transfers_sensor_and_hazard_evidence_without_control_artifacts(self):
        for scenario, phrase in (("fire", "Fire alarm sounded"), ("rain", "Wipers have started")):
            with self.subTest(scenario=scenario):
                sdk, requests, actors, readings = self.sdk(scenario)
                workflow = handoff.build_workflow(sdk, readings=readings, sensor_delay=0, fire_delay=0, rain_delay=0)
                await workflow.run(handoff.PROMPT)
                motor_request = next(r for r, a in zip(requests, actors) if a == "MotorsAgent")
                self.assertIn(phrase, json.dumps(motor_request["messages"][1:]))
                self.assertIn("read_temperature", json.dumps(motor_request["messages"]))
                self.assertNotIn('"tool_call_id"', json.dumps(motor_request["messages"]))
                self.assertTrue(any(t["function"]["name"] == "stop" for t in motor_request["tools"]))
                self.assertFalse(any(t["function"]["name"] == "sound_alarm" for t in motor_request["tools"]))

    async def test_per_edge_reasons_not_router_instructions_and_declared_topology(self):
        sdk, _, _, _ = self.sdk()
        for reasons in (False, True):
            with self.subTest(reasons=reasons):
                workflow = handoff.build_workflow(sdk, reasons=reasons)
                executor = workflow.executors["EnvironmentAgent"]
                description_by_name = {t.name: t.description for t in executor.agent.default_options["tools"]
                                       if t.name.startswith("handoff_to_")} if hasattr(executor, "agent") else {
                    t.name: t.description for t in executor._agent.default_options["tools"] if t.name.startswith("handoff_to_")}
                self.assertEqual(set(description_by_name), {"handoff_to_"+name for name in handoff.ROUTING_REASONS})
                instructions = (executor.agent if hasattr(executor, "agent") else executor._agent).default_options["instructions"]
                if reasons:
                    for target, reason in handoff.ROUTING_REASONS.items():
                        self.assertEqual(description_by_name["handoff_to_"+target], reason)
                        self.assertNotIn(reason, instructions)
                else:
                    for reason in handoff.ROUTING_REASONS.values():
                        self.assertIn(reason, instructions)
                diagram = WorkflowViz(workflow).to_mermaid()
                for actor in ("EnvironmentAgent", "FireDetectorAgent", "RainDetectorAgent", "MotorsAgent"):
                    self.assertIn(actor, diagram)

    async def test_specialists_require_actual_handoff_tool_not_prose(self):
        sdk, _, _, _ = self.sdk()
        for reasons in (False, True):
            agents = handoff.build_agents(sdk, reasons=reasons)
            for name in ("FireDetectorAgent", "RainDetectorAgent"):
                instructions = agents[name].default_options["instructions"]
                self.assertIn("invoke the", instructions)
                self.assertIn("handoff_to_MotorsAgent tool", instructions)
                self.assertIn("ordinary text does not", instructions)
            router = agents["EnvironmentAgent"].default_options["instructions"]
            self.assertIn("read_temperature and read_droplet_level", router)
            self.assertIn("Invoke the selected handoff_to_* tool", router)
            motors = agents["MotorsAgent"].default_options["instructions"]
            self.assertIn('headed "Completed tool results"', motors)
            self.assertIn("invoke the stop tool", motors)
            self.assertIn("Do not execute the obstacle-avoidance", motors)

    async def test_motor_authority_and_tool_domains_separated(self):
        sdk, _, _, _ = self.sdk()
        agents = handoff.build_agents(sdk)
        names = {name: {t.name for t in agent.default_options["tools"]} for name, agent in agents.items()}
        self.assertEqual(names["MotorsAgent"], {"turn_right", "turn_left", "forward", "backward", "stop"})
        self.assertEqual(names["RainDetectorAgent"], {"start_wipers", "stop_wipers"})
        self.assertEqual(names["FireDetectorAgent"], {"start_water_sprinkle", "stop_water_sprinkle", "sound_alarm"})
        self.assertEqual(len(names["EnvironmentAgent"]), 6)
        self.assertTrue(all("stop" not in values for key, values in names.items() if key != "MotorsAgent"))

    async def test_evidence_keeps_handoff_result_last_and_does_not_copy_routing_signal(self):
        last = Message("tool", [Content.from_function_result("route", result={"handoff_to": "MotorsAgent"})])
        response = AgentResponse(messages=[Message("assistant", [
            Content.from_function_call("alarm", "sound_alarm", arguments={}),
            Content.from_function_call("route", "handoff_to_MotorsAgent", arguments={}),
        ]), Message("tool", [Content.from_function_result("alarm", result="Fire alarm sounded.")]), last])
        handoff.ToolEvidenceMiddleware.add_evidence(response, "FireDetectorAgent")
        self.assertIs(response.messages[-1], last)
        self.assertIn("sound_alarm: Fire alarm sounded.", response.messages[-2].text)
        self.assertNotIn("handoff_to", response.messages[-2].text)
        self.assertEqual(response.messages[-2].author_name, "FireDetectorAgent")

    async def test_missing_handoff_pauses_instead_of_auto_executing_motors(self):
        sdk, requests, actors, readings = self.sdk(no_handoff=True)
        workflow = handoff.build_workflow(sdk, readings=readings, sensor_delay=0)
        result = await workflow.run(handoff.PROMPT)
        self.assertEqual(len(result.get_request_info_events()), 1)
        self.assertEqual(actors, ["EnvironmentAgent"])
        self.assertEqual(len(requests), 1)
        self.assertNotIn("MOTORS:", self.output.getvalue())

    async def test_native_streaming_route_outputs_and_console(self):
        sdk, requests, actors, readings = self.sdk("fire", stream=True)
        workflow = handoff.build_workflow(sdk, readings=readings, sensor_delay=0, fire_delay=0)
        result = await handoff.run_workflow(workflow)
        self.assertEqual(result.get_request_info_events(), [])
        self.assertIn("[HANDOFF] EnvironmentAgent", self.output.getvalue())
        self.assertIn("[WORKFLOW OUTPUT] Hazard detected; stopped.", self.output.getvalue())
        self.assertIn("Fire alarm sounded", json.dumps(next(r["messages"] for r, a in zip(requests, actors) if a == "MotorsAgent")))
        self.assertEqual(actors.count("MotorsAgent"), 2)

    async def test_reading_scenarios_and_sensor_ranges_preserve_source(self):
        self.assertEqual(handoff.SensorReadings.scenario("random"), handoff.SensorReadings())
        self.assertEqual(handoff.SensorReadings.scenario("fire").temperature, 85)
        self.assertEqual(handoff.SensorReadings.scenario("rain").droplet_level, handoff.DropletLevel.HIGH)
        with self.assertRaises(ValueError):
            handoff.SensorReadings.scenario("invalid")
        sensors = handoff.EnvironmentTools(handoff.SensorReadings(temperature=0, humidity=0, infrared_radiation=0,
                                                                 wind_speed=0, wind_direction=handoff.Direction.NORTH,
                                                                 droplet_level=handoff.DropletLevel.NONE), delay=0)
        self.assertEqual(await sensors.read_temperature(), 0)
        self.assertEqual(await sensors.read_humidity(), 0)
        self.assertEqual(await sensors.read_infrared_radiation(), 0)
        self.assertEqual(await sensors.read_wind_speed(), 0)
        self.assertEqual(await sensors.read_wind_direction(), handoff.Direction.NORTH)
        self.assertEqual(await sensors.read_droplet_level(), handoff.DropletLevel.NONE)
        random_sensors = handoff.EnvironmentTools(delay=0)
        with patch.object(handoff, "randrange", side_effect=[-20, 99, 99, 99]) as random:
            self.assertEqual(await random_sensors.read_temperature(), -20)
            self.assertEqual(await random_sensors.read_humidity(), 99)
            self.assertEqual(await random_sensors.read_infrared_radiation(), 99)
            self.assertEqual(await random_sensors.read_wind_speed(), 99)
        self.assertEqual(random.call_args_list[0].args, (-20, 100))

    async def test_detector_functions_return_source_phrases_and_validate_enum(self):
        detectors = handoff.DetectorTools(fire_delay=0, rain_delay=0)
        self.assertEqual(await detectors.sound_alarm(), "Fire alarm sounded.")
        self.assertIn("Water sprinkler started", await detectors.start_water_sprinkle())
        self.assertEqual(await detectors.stop_water_sprinkle(), "Water sprinkler stopped.")
        self.assertEqual(await detectors.start_wipers(handoff.DropletLevel.HIGH), "Wipers have started for droplet level: High.")
        self.assertEqual(await detectors.stop_wipers(), "Wipers have stopped.")
        tool = next(t for t in detectors.rain_tools() if t.name == "start_wipers")
        with self.assertRaises(TypeError):
            await tool.invoke(arguments={"droplet_level": "Flooded"})

    async def test_termination_requires_latest_motor_message_not_text_substring(self):
        self.assertFalse(handoff.motor_completed([]))
        self.assertFalse(handoff.motor_completed([Message("assistant", ["MotorsAgent approved"], author_name="EnvironmentAgent")]))
        self.assertTrue(handoff.motor_completed([Message("assistant", ["Finished"], author_name="MotorsAgent")]))
        self.assertFalse(handoff.motor_completed([Message("assistant", ["Finished"], author_name="MotorsAgent"), Message("user", ["continue"])]))

    async def test_both_entrypoints_provider_and_sdk_cleanup(self):
        for name, reasons in (("HandoffOrchestration", False), ("HandoffOrchestrationAndHandoffReasons", True)):
            with self.subTest(sample=name):
                spec = importlib.util.spec_from_file_location("handoff_"+name, ROOT / "14.Handoff" / name / "main.py")
                sample = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sample)
                sdk = Mock()
                sdk.__aenter__ = AsyncMock(return_value=sdk)
                sdk.__aexit__ = AsyncMock(return_value=False)
                with patch.object(sample, "create_sdk_client", return_value=sdk) as factory, \
                     patch.object(sample, "build_workflow", return_value="workflow") as builder, \
                     patch.object(sample, "run_workflow", new=AsyncMock(return_value="result")) as run:
                    self.assertEqual(await sample.main("azure", "rain"), "result")
                factory.assert_called_once_with("azure")
                self.assertEqual(builder.call_args.kwargs["reasons"], reasons)
                self.assertEqual(builder.call_args.kwargs["readings"].droplet_level, handoff.DropletLevel.HIGH)
                run.assert_awaited_once_with("workflow")
                sdk.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
