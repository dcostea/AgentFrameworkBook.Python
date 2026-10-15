"""Offline native workflow, manager, HITL, and provider regression tests for chapter 13."""

import importlib.util
import inspect
import io
import json
import os
import re
import sys
import unittest
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with patch("dotenv.load_dotenv", return_value=False):
    from agent_framework import (AgentExecutor, AgentResponse, AgentResponseUpdate, AgentSession,
                                 Content, Message, ResponseStream, SupportsAgentRun, tool)
    from agent_framework.orchestrations import BaseGroupChatOrchestrator, GroupChatBuilder, GroupChatState
    from shared import chat_clients, group_chat_examples as gc, motor_tools

PROJECTS = ("GroupChatOrchestration", "GroupChatOrchestrationWithCustomManager", "GroupChatOrchestrationWithHitl")
ENV = {"OPENAI_API_KEY": "offline", "OPENAI_CHAT_MODEL_ID": "offline-model",
       "AZURE_OPENAI_ENDPOINT": "https://azure.invalid", "AZURE_OPENAI_API_KEY": "offline",
       "AZURE_OPENAI_DEPLOYMENT_NAME": "offline-deployment"}


def message(author, text):
    return Message("assistant", [text], author_name=author)


def load_sample(name):
    spec = importlib.util.spec_from_file_location(f"group_chat_{name}", ROOT / "13.GroupChat" / name / "main.py")
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


class ScriptedAgent:
    def __init__(self, name, responses):
        self.id = self.name = name
        self.description = name
        self.responses = list(responses)
        self.inputs = []

    def create_session(self, *, session_id=None):
        return AgentSession(session_id=session_id)

    def get_session(self, service_session_id, *, session_id=None):
        return AgentSession(service_session_id=service_session_id, session_id=session_id)

    def run(self, messages=None, *, stream=False, session=None, **kwargs):
        self.inputs.append(list(messages or []))
        index = len(self.inputs) - 1
        if index >= len(self.responses):
            raise AssertionError(f"Unexpected turn for {self.name}")
        value = self.responses[index]
        if isinstance(value, Exception):
            raise value
        response = value if isinstance(value, AgentResponse) else AgentResponse(messages=[message(self.name, value)])
        if not stream:
            async def complete():
                return response
            return complete()

        async def updates():
            for msg in response.messages:
                yield AgentResponseUpdate(contents=msg.contents, role=msg.role, author_name=msg.author_name)
        return ResponseStream(updates(), finalizer=lambda _: response)


def completion(text="", calls=None):
    return {"id": "chatcmpl-offline", "object": "chat.completion", "created": 1, "model": "offline-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": {
                "role": "assistant", "content": None if calls else text,
                **({"tool_calls": [{"id": cid, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}} for cid, name, args in calls]} if calls else {})}}]}


def streamed_completion(text="", calls=None):
    delta = {"role": "assistant", "content": text or None}
    if calls:
        delta["tool_calls"] = [{"index": index, "id": cid, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args)}} for index, (cid, name, args) in enumerate(calls)]
    chunks = [{"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 1,
               "model": "offline-model", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
              {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 1,
               "model": "offline-model", "choices": [{"index": 0, "delta": {},
                  "finish_reason": "tool_calls" if calls else "stop"}]}]
    return httpx.Response(200, content=("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                                       + "data: [DONE]\n\n").encode(),
                          headers={"Content-Type": "text/event-stream"})


class GroupChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, ENV, clear=True))
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request",
                                      new=AsyncMock(side_effect=AssertionError("Real network forbidden"))))
        self.enterContext(patch.object(httpx.HTTPTransport, "handle_request",
                                      side_effect=AssertionError("Real network forbidden")))
        self.enterContext(patch.object(motor_tools, "_DELAY_SECONDS", 0))
        self.enterContext(patch("logging.Logger.error"))
        self.enterContext(patch("logging.Logger.warning"))
        self.enterContext(patch("logging.Logger.exception"))
        self.stdout = io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))

    def sdk(self, responses, *, provider="openai", legacy=False):
        requests = []
        responses = list(responses)

        def handle(request):
            data = json.loads(request.content)
            requests.append((request, data))
            if len(requests) > len(responses):
                raise AssertionError("Unexpected provider request")
            value = responses[len(requests) - 1]
            if isinstance(value, httpx.Response):
                return value
            text, calls = value if isinstance(value, tuple) else (value, None)
            return streamed_completion(text, calls) if data.get("stream") else httpx.Response(200, json=completion(text, calls))

        if legacy:
            os.environ["AZURE_OPENAI_API_VERSION"] = "2025-04-01-preview"
        sdk = chat_clients.create_sdk_client(provider)
        old_client = sdk._client
        sdk._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(old_client.aclose)
        self.addAsyncCleanup(sdk.close)
        return sdk, requests

    def execution_workflow(self, motors, navigator, *, human=None, cap=10):
        participants = [motors, navigator]
        if human is not None:
            participants.append(gc.HumanAgentExecutor(human))
        manager = gc.ApprovedTerminationManager(tuple(p.id if isinstance(p, gc.Executor) else p.name
                                                      for p in participants), cap,
                                               "HumanAgent" if human is not None else "NavigatorAgent")
        workflow = GroupChatBuilder(participants=participants, selection_func=manager.select_next,
                                    termination_condition=manager.should_terminate, max_rounds=cap,
                                    output_from=["MotorsAgent"], intermediate_output_from="all_other").build()
        return gc.GroupChatSample(workflow, manager, "MotorsAgent", gc.MISSION)

    def investigation_workflow(self, environment, safety, maintenance=None, *, cap=10):
        # Scripted reports carry real, fresh call/result envelopes, just like model tool runs.
        environment.responses = [self.observation_result(*map(int, re.findall(r"-?\d+", response)[:2]), index=index)
                                 if isinstance(response, str) else response
                                 for index, response in enumerate(environment.responses)]
        agents = [environment, maintenance or ScriptedAgent("MaintenanceAgent", []), safety]
        executors = [AgentExecutor(agent, id=agent.name) for agent in agents]
        manager = gc.InvestigationManager(executors, maximum_iteration_count=cap)
        workflow = GroupChatBuilder(participants=executors, orchestrator=manager, output_from=["SafetyAgent"],
                                    intermediate_output_from="all_other").build()
        return gc.GroupChatSample(workflow, manager, "SafetyAgent", gc.INVESTIGATION_PROMPT)

    def observation_result(self, temperature=20, humidity=10, *, index=0):
        return AgentResponse(messages=[Message("assistant", [
            Content.from_function_call(f"t-{index}", "read_temperature", arguments={}),
            Content.from_function_call(f"h-{index}", "read_humidity", arguments={})]),
            Message("tool", [Content.from_function_result(f"t-{index}", result=temperature),
                             Content.from_function_result(f"h-{index}", result=humidity)]),
            message("EnvironmentAgent", f"Observation: temperature = {temperature} Celsius; humidity = {humidity}%.")])

    def maintenance_result(self, *, success=True, result="All sensors have been calibrated."):
        return AgentResponse(messages=[Message("tool", [Content.from_function_result(
            "calibration", result=result, exception=None if success else "calibration failed")]),
            message("MaintenanceAgent", "Calibration complete." if success else "Calibration failed.")])

    async def test_decision_keywords_and_typed_json_are_exact(self):
        for text, expected in (("APPROVED: safe", gc.ExecutionDecision.APPROVED),
                               (" denied: mention APPROVED", gc.ExecutionDecision.DENIED),
                               ("EXECUTED: done", gc.ExecutionDecision.EXECUTED),
                               ('{"decision":"APPROVED","reason":"ok"}', gc.ExecutionDecision.APPROVED)):
            self.assertEqual(gc.parse_decision(text), expected)
        for text in ("NOT APPROVED", "UNEXECUTED", "Reason: APPROVED", "APPROVED eventually", "HUMAN APPROVED: ok",
                     '{"reason":"APPROVED"}', "{bad json}", "[]"):
            self.assertIsNone(gc.parse_decision(text))
        self.assertEqual(gc.parse_decision("HUMAN APPROVED: okay", human=True), gc.ExecutionDecision.APPROVED)
        self.assertEqual(gc.parse_decision("CALIBRATION: mentions APPROVED", gc.SafetyDecision), gc.SafetyDecision.CALIBRATION)
        self.assertIsNone(gc.parse_decision("APPROVED: okay", human=True))
        self.assertEqual(gc.ExecutionVerdict(decision="EXECUTED", reason="done").decision, gc.ExecutionDecision.EXECUTED)

    async def test_termination_requires_latest_authoritative_approval_and_motors(self):
        manager = gc.ApprovedTerminationManager()
        approved = message("NavigatorAgent", "APPROVED: good")
        denied = message("NavigatorAgent", "DENIED: an APPROVED path would differ")
        executed = message("MotorsAgent", "EXECUTED: done")
        self.assertTrue(manager.should_terminate([approved, executed]))
        for history in ([executed], [approved, denied, executed],
                        [message("NavigatorAgent", "NOT APPROVED"), executed],
                        [approved, message("MotorsAgent", "UNEXECUTED")],
                        [approved, message("HumanAgent", "EXECUTED: no")],
                        [Message("user", ["APPROVED"], author_name="NavigatorAgent"), executed]):
            self.assertFalse(manager.should_terminate(history))

    async def test_round_robin_actor_order_and_turn_limit(self):
        manager = gc.ApprovedTerminationManager()
        registry = OrderedDict((name, name) for name in manager.participants)
        self.assertEqual([manager.select_next(GroupChatState(i, registry, [])) for i in range(5)],
                         ["MotorsAgent", "NavigatorAgent", "MotorsAgent", "NavigatorAgent", "MotorsAgent"])
        self.assertEqual(manager.iteration_count, 5)
        self.assertEqual(manager.last_agent, "MotorsAgent")
        manager.iteration_count = 10
        self.assertTrue(manager.should_terminate([]))
        for cap in (0, -1):
            with self.assertRaises(ValueError):
                gc.ApprovedTerminationManager(maximum_iteration_count=cap)
        with self.assertRaises(ValueError):
            gc.ApprovedTerminationManager(participants=("MotorsAgent",))

    async def test_real_round_robin_workflow_approval_last_actor_and_stream_events(self):
        motors = ScriptedAgent("MotorsAgent", ["Proposed path", "EXECUTED: avoided tree"])
        navigator = ScriptedAgent("NavigatorAgent", ["APPROVED: use 45 degrees"])
        sample = self.execution_workflow(motors, navigator)
        result = await gc.run_streaming(sample)
        self.assertEqual(sample.manager.iteration_count, 3)
        self.assertEqual(sample.last_response, "EXECUTED: avoided tree")
        self.assertEqual(len(result.get_outputs()), 3)  # Two motor turns and terminal orchestrator output.
        self.assertIn("[NavigatorAgent]", self.stdout.getvalue())
        self.assertIn("[WORKFLOW OUTPUT] EXECUTED: avoided tree", self.stdout.getvalue())
        self.assertEqual(gc.latest_message(motors.inputs[-1], "NavigatorAgent").text, "APPROVED: use 45 degrees")

    async def test_real_workflow_denial_revises_and_is_bounded(self):
        motors = ScriptedAgent("MotorsAgent", ["Proposal 1", "Proposal 2", "Proposal 3", "Proposal 4", "Proposal 5"])
        navigator = ScriptedAgent("NavigatorAgent", ["DENIED: use 45 degrees"] * 5)
        sample = self.execution_workflow(motors, navigator)
        await sample.workflow.run(sample.prompt)
        self.assertEqual((len(motors.inputs), len(navigator.inputs), sample.manager.iteration_count), (5, 5, 10))
        self.assertEqual(sample.last_response, "Proposal 5")
        self.assertEqual(gc.latest_message(motors.inputs[1], "NavigatorAgent").text, "DENIED: use 45 degrees")

    async def test_human_agent_protocol_sessions_yes_no_feedback_and_eof(self):
        proposal = [message("MotorsAgent", "Turn 45 then forward 5")]
        for answer, decision in (("Y", "APPROVED"), ("yes", "APPROVED"), ("n", "DENIED"), ("", "DENIED"),
                                 ("Use 30 degrees", "DENIED")):
            read = Mock(return_value=answer)
            human = gc.HumanAgent(read)
            self.assertIsInstance(human, SupportsAgentRun)
            self.assertIsInstance(human.create_session(session_id="local"), AgentSession)
            self.assertEqual(human.get_session("remote").service_session_id, "remote")
            response = await human.run(proposal)
            self.assertTrue(response.text.startswith(f"HUMAN {decision}:"))
            self.assertEqual(response.messages[0].author_name, "HumanAgent")
            self.assertIn("Turn 45 then forward 5", read.call_args.args[0])
            if answer == "Use 30 degrees":
                self.assertIn(answer, response.text)
        human = gc.HumanAgent(Mock(side_effect=EOFError))
        self.assertTrue((await human.run(proposal)).text.startswith("HUMAN DENIED:"))
        self.assertTrue((await gc.HumanAgent(lambda _: "y").run([])).text.startswith("HUMAN DENIED:"))

    async def test_human_stream_and_already_executed_does_not_prompt(self):
        read = Mock(side_effect=AssertionError("Already executed must not prompt"))
        human = gc.HumanAgent(read)
        stream = human.run([message("MotorsAgent", "EXECUTED: done")], stream=True)
        updates = [update async for update in stream]
        response = await stream.get_final_response()
        self.assertEqual(updates[0].author_name, "HumanAgent")
        self.assertEqual(response.text, "HUMAN APPROVED: sequence approved already.")
        read.assert_not_called()
        false_executed = gc.HumanAgent(Mock(return_value="n"))
        self.assertTrue((await false_executed.run([message("MotorsAgent", "NOT EXECUTED")])).text.startswith("HUMAN DENIED:"))

    async def test_hitl_real_workflow_human_yes_overrides_navigator_no(self):
        motors = ScriptedAgent("MotorsAgent", ["Proposed path", "EXECUTED: human-authorized"])
        navigator = ScriptedAgent("NavigatorAgent", ["DENIED: advisory suggestion"])
        read = Mock(return_value="yes")
        sample = self.execution_workflow(motors, navigator, human=gc.HumanAgent(read))
        await gc.run_streaming(sample)
        self.assertEqual(sample.manager.iteration_count, 4)
        self.assertEqual(sample.last_response, "EXECUTED: human-authorized")
        self.assertEqual(gc.parse_decision(gc.latest_message(motors.inputs[-1], "HumanAgent").text, human=True),
                         gc.ExecutionDecision.APPROVED)
        self.assertIn("[HumanAgent]", self.stdout.getvalue())
        read.assert_called_once()

    async def test_hitl_denied_custom_feedback_and_eof_stay_within_ten_turns(self):
        motors = ScriptedAgent("MotorsAgent", ["Proposal"] * 4)
        navigator = ScriptedAgent("NavigatorAgent", ["APPROVED: advisory"] * 3)
        read = Mock(side_effect=["Use 30 degrees", "no", EOFError()])
        sample = self.execution_workflow(motors, navigator, human=gc.HumanAgent(read))
        await sample.workflow.run(sample.prompt)
        self.assertEqual(sample.manager.iteration_count, 10)
        self.assertEqual(read.call_count, 3)
        self.assertIn("Use 30 degrees", gc.latest_message(motors.inputs[1], "HumanAgent").text)
        self.assertTrue(gc.latest_message(motors.inputs[-1], "HumanAgent").text.startswith("HUMAN DENIED:"))
        self.assertNotIn("EXECUTED", sample.last_response)

    async def test_investigation_safe_six_turn_route_and_explicit_integer_state(self):
        environment = ScriptedAgent("EnvironmentAgent", ["Observation: temperature = 60 Celsius; humidity = 80%."] * 2)
        maintenance = ScriptedAgent("MaintenanceAgent", [self.maintenance_result()])
        safety = ScriptedAgent("SafetyAgent", ["CALIBRATION: second observation", "CALIBRATION: calibrate",
                                                "APPROVED: actual successful calibration"])
        sample = self.investigation_workflow(environment, safety, maintenance)
        self.assertIsInstance(sample.manager, BaseGroupChatOrchestrator)
        await sample.workflow.run(sample.prompt)
        self.assertEqual((sample.manager.observation_count, sample.manager.iteration_count), (2, 6))
        self.assertIs(type(sample.manager.observation_count), int)
        self.assertTrue(sample.manager.calibrated)
        self.assertEqual(sample.manager.last_agent, "SafetyAgent")
        self.assertEqual(sample.last_response, "APPROVED: actual successful calibration")
        state = [m for m in safety.inputs[-1] if m.author_name == "InvestigationManager"][-1]
        self.assertEqual(state.role, "user")
        self.assertIn("Completed observations: 2", state.text)
        self.assertIn("Calibration completed: True", state.text)
        self.assertTrue(any(c.type == "function_result" for m in sample.manager.raw_history for c in m.contents))
        self.assertFalse(any(c.type == "function_result" for m in sample.manager.history for c in m.contents))

    async def test_investigation_extreme_first_observation_denies_immediately(self):
        environment = ScriptedAgent("EnvironmentAgent", ["Observation: temperature = 91 Celsius; humidity = 30%."])
        safety = ScriptedAgent("SafetyAgent", ["DENIED: temperature 91 exceeds 90; not APPROVED"])
        sample = self.investigation_workflow(environment, safety)
        await sample.workflow.run(sample.prompt)
        self.assertEqual(sample.manager.iteration_count, 2)
        self.assertEqual(sample.manager.observation_count, 1)
        self.assertFalse(sample.manager.calibrated)

    async def test_investigation_second_observation_checks_prior_data_role(self):
        environment = ScriptedAgent("EnvironmentAgent", ["Observation: temperature = 70 Celsius; humidity = 10%.",
                                                          "Observation: temperature = 20 Celsius; humidity = 10%."])
        safety = ScriptedAgent("SafetyAgent", ["CALIBRATION: collect second", "DENIED: first temperature 70 exceeds 60"])
        sample = self.investigation_workflow(environment, safety)
        await sample.workflow.run(sample.prompt)
        self.assertEqual(sample.manager.iteration_count, 4)
        self.assertIn("70", safety.inputs[0][-2].text)
        self.assertTrue(any("20 Celsius" in m.text for m in safety.inputs[1]))
        self.assertEqual(sample.last_response, "DENIED: first temperature 70 exceeds 60")

    async def test_failed_calibration_retries_without_false_approval_to_cap(self):
        environment = ScriptedAgent("EnvironmentAgent", ["Observation: temperature = 20 Celsius; humidity = 10%."] * 2)
        safety = ScriptedAgent("SafetyAgent", ["CALIBRATION: second", "CALIBRATION: maintenance",
                                                "APPROVED: model claims done", "CALIBRATION: retry", "APPROVED: unverified"])
        maintenance = ScriptedAgent("MaintenanceAgent", [self.maintenance_result(success=False)] * 3)
        sample = self.investigation_workflow(environment, safety, maintenance)
        await sample.workflow.run(sample.prompt)
        self.assertFalse(sample.manager.calibrated)
        self.assertEqual(sample.manager.observation_count, 2)
        self.assertEqual(sample.manager.iteration_count, 10)
        self.assertEqual(len(maintenance.inputs), 3)
        self.assertEqual(len(environment.inputs), 2)

    async def test_calibration_claim_null_or_failed_result_is_not_evidence(self):
        for value in (AgentResponse(messages=[message("MaintenanceAgent", "Calibration complete")]),
                      self.maintenance_result(result=None), self.maintenance_result(success=False)):
            sample = self.investigation_workflow(ScriptedAgent("EnvironmentAgent", []), ScriptedAgent("SafetyAgent", []))
            sample.manager.last_agent = "MaintenanceAgent"
            response = gc.AgentExecutorResponse("MaintenanceAgent", value, value.messages)
            sample.manager._process_participant_response(response)
            self.assertFalse(sample.manager.calibrated)
        # A successful sensor result is not a successful current Maintenance result.
        sample.manager.last_agent = "EnvironmentAgent"
        sample.manager._process_participant_response(gc.AgentExecutorResponse("EnvironmentAgent", self.maintenance_result(), []))
        self.assertFalse(sample.manager.calibrated)
        self.assertEqual(sample.manager.observation_count, 0)
        manager = sample.manager
        manager.last_agent = "EnvironmentAgent"
        valid = self.observation_result(index=9)
        manager._process_participant_response(gc.AgentExecutorResponse("EnvironmentAgent", valid, []))
        self.assertEqual(manager.observation_count, 1)
        for invalid in (valid, AgentResponse(messages=[message("EnvironmentAgent", "Observation: same old readings")]),
                        self.observation_result(temperature="20.5", index=10),
                        self.observation_result(temperature=True, index=11)):
            manager._process_participant_response(gc.AgentExecutorResponse("EnvironmentAgent", invalid, []))
            self.assertEqual(manager.observation_count, 1)
        self.assertEqual(manager.observations, [(20, 10)])
        partial = self.observation_result(index=12)
        partial.messages[1].contents = partial.messages[1].contents[:1]
        manager._process_participant_response(gc.AgentExecutorResponse("EnvironmentAgent", partial, []))
        replay_partial = self.observation_result(index=12)
        manager._process_participant_response(gc.AgentExecutorResponse("EnvironmentAgent", replay_partial, []))
        self.assertEqual(manager.observation_count, 1)
        # Match the live failure: a repeated report without tool calls is not observation two.
        environment = ScriptedAgent("EnvironmentAgent", [self.observation_result(index=15),
            AgentResponse(messages=[message("EnvironmentAgent", "Observation: temperature = 20 Celsius; humidity = 10%.")])])
        safety = ScriptedAgent("SafetyAgent", ["CALIBRATION: second", "CALIBRATION: still need second"])
        repeated = self.investigation_workflow(environment, safety, cap=4)
        await repeated.workflow.run(repeated.prompt)
        self.assertEqual(repeated.manager.observation_count, 1)
        self.assertIn("Unverified Environment report", "\n".join(m.text for m in safety.inputs[-1]))
        self.assertIn("Completed observations: 1", "\n".join(m.text for m in safety.inputs[-1]))

    async def test_calibrated_nonterminal_safety_routes_back_to_environment(self):
        environment = ScriptedAgent("EnvironmentAgent", ["Observation: 20 C, 10%"] * 3)
        safety = ScriptedAgent("SafetyAgent", ["CALIBRATION: second", "CALIBRATION: maintenance",
                                                "CALIBRATION: investigate cycle", "DENIED: end investigation"])
        maintenance = ScriptedAgent("MaintenanceAgent", [self.maintenance_result()])
        sample = self.investigation_workflow(environment, safety, maintenance)
        await sample.workflow.run(sample.prompt)
        self.assertEqual(sample.manager.iteration_count, 8)
        self.assertEqual(sample.manager.observation_count, 3)
        self.assertEqual(len(maintenance.inputs), 1)
        self.assertTrue(sample.manager.calibrated)

    async def test_investigation_only_safety_decisions_terminate(self):
        sample = self.investigation_workflow(ScriptedAgent("EnvironmentAgent", []), ScriptedAgent("SafetyAgent", []))
        manager = sample.manager
        manager.last_agent = "SafetyAgent"
        for text in ("CALIBRATION: mention APPROVED", "NOT APPROVED", "Reason: DENIED", "APPROVED: missing calibration"):
            self.assertFalse(manager.should_terminate([message("SafetyAgent", text)]))
        manager.calibrated = True
        self.assertFalse(manager.should_terminate([message("SafetyAgent", "APPROVED: no verified observations")]))
        manager.observation_count = 2
        self.assertTrue(manager.should_terminate([message("SafetyAgent", "APPROVED: calibrated")]))
        manager.last_agent = "EnvironmentAgent"
        self.assertFalse(manager.should_terminate([message("EnvironmentAgent", "DENIED: not safety")]))
        with self.assertRaises(ValueError):
            gc.InvestigationManager([], maximum_iteration_count=0)

    async def test_provider_execution_tools_off_until_approval_and_latest_history(self):
        sdk, requests = self.sdk(["Proposal: turn 45 then forward 5", "DENIED: use 30", "Revised proposal: 30",
                                  "APPROVED: revised path", ("", [("move", "forward", {"distance": 5})]),
                                  "EXECUTED: forward 5"])
        sample = gc.build_execution(sdk)
        await sample.workflow.run(sample.prompt)
        self.assertEqual(sample.manager.iteration_count, 5)
        self.assertEqual(len(requests), 6)
        for index in (0, 2):
            data = requests[index][1]
            self.assertNotIn("tool_choice", data)  # Adapter omits this when no tools are sent.
            self.assertFalse(data.get("tools"))
            planning = data["messages"][-1]["content"]
            self.assertIn("Planning/revision phase", planning)
            self.assertIn('"angle"', planning)
            self.assertIn('"distance"', planning)
            self.assertIn("NOT permission", planning)
        self.assertIn("Execution phase", requests[4][1]["messages"][-1]["content"])
        self.assertEqual(requests[4][1]["tool_choice"], "auto")
        self.assertEqual({t["function"]["name"] for t in requests[4][1]["tools"]},
                         {"forward", "backward", "turn_left", "turn_right", "stop"})
        self.assertEqual(self.stdout.getvalue().count("MOTORS:"), 1)
        self.assertEqual(sample.last_response, "EXECUTED: forward 5")
        self.assertTrue(any("use 30" in str(m) for m in requests[2][1]["messages"]))
        self.assertTrue(any(m["role"] == "tool" for m in requests[5][1]["messages"]))
        self.assertFalse(any(m["role"] == "tool" for m in requests[3][1]["messages"]))

    async def test_provider_hitl_no_has_no_motor_tools_and_no_motor_calls(self):
        sdk, requests = self.sdk(["Proposed path", "APPROVED: navigator", "Revised path", "APPROVED: navigator"])
        read = Mock(side_effect=["n", EOFError()])
        sample = gc.build_execution(sdk, hitl=True, read_input=read, maximum_iteration_count=6)
        await sample.workflow.run(sample.prompt)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(sample.manager.iteration_count, 6)
        self.assertEqual(self.stdout.getvalue().count("MOTORS:"), 0)
        for index in (0, 2):
            self.assertFalse(requests[index][1].get("tools"))
            self.assertNotIn("tool_choice", requests[index][1])

    async def test_provider_hitl_yes_executes_despite_advisory_denial(self):
        sdk, requests = self.sdk(["Proposed path", "DENIED: advisory", ("", [("halt", "stop", {})]), "EXECUTED: stopped"])
        sample = gc.build_execution(sdk, hitl=True, read_input=lambda _: "y")
        await gc.run_streaming(sample)
        self.assertEqual(sample.manager.iteration_count, 4)
        self.assertEqual(requests[2][1]["tool_choice"], "auto")
        self.assertEqual(self.stdout.getvalue().count("MOTORS:"), 1)
        self.assertEqual(sample.last_response, "EXECUTED: stopped")

    async def test_real_provider_investigation_tools_integer_results_and_text_only_handoff(self):
        async def temperature() -> int:
            return 60
        async def humidity() -> int:
            return 80
        async def calibration() -> str:
            return "All sensors have been calibrated."
        counts = {"temperature": 0, "humidity": 0}
        original_temperature, original_humidity = temperature, humidity
        async def counted_temperature() -> int:
            counts["temperature"] += 1
            return await original_temperature()
        async def counted_humidity() -> int:
            counts["humidity"] += 1
            return await original_humidity()
        sensors = [("t1", "temperature", {}), ("h1", "humidity", {})]
        sdk, requests = self.sdk([("", sensors), "Observation: temperature = 60 Celsius; humidity = 80%.",
                                  "CALIBRATION: second", ("", [("t2", "temperature", {}), ("h2", "humidity", {})]),
                                  "Observation: temperature = 60 Celsius; humidity = 80%.", "CALIBRATION: calibrate",
                                  ("", [("c", "calibration", {})]), "Calibration complete.", "APPROVED: calibrated"])
        sample = gc.build_investigation(sdk, sensor_tools=[tool(counted_temperature, name="temperature"),
                                                         tool(counted_humidity, name="humidity")],
                                        calibration_tool=tool(calibration))
        await gc.run_streaming(sample)
        self.assertEqual(sample.manager.iteration_count, 6)
        self.assertTrue(sample.manager.calibrated)
        self.assertEqual(sample.manager.observation_count, 2)
        self.assertEqual(counts, {"temperature": 2, "humidity": 2})
        for index in (0, 3):
            self.assertEqual(requests[index][1]["tool_choice"], "required")
            self.assertEqual({entry["function"]["name"] for entry in requests[index][1]["tools"]},
                             {"temperature", "humidity"})
            self.assertIn("NEW independent observation", requests[index][1]["messages"][-1]["content"])
        for index in (1, 4):
            self.assertFalse(requests[index][1].get("tools"))
        tool_results = [m for m in requests[1][1]["messages"] if m["role"] == "tool"]
        self.assertEqual([json.loads(m["content"]) for m in tool_results], [60, 80])
        for index in (2, 5, 8):
            self.assertFalse(any(m["role"] == "tool" or m.get("tool_calls") for m in requests[index][1]["messages"]))
        self.assertIn("Calibration completed: True", str(requests[-1][1]["messages"]))
        self.assertNotIn("forward", str(requests[6][1].get("tools")))
        # A provider may request one sensor at a time; require only the missing one next.
        sequential_sdk, sequential_requests = self.sdk([
            ("", [("t3", "temperature", {})]), ("", [("h3", "humidity", {})]),
            "Observation: temperature = 60 Celsius; humidity = 80%."], provider="openai")
        sequential = gc.build_investigation(sequential_sdk,
                                            sensor_tools=[tool(counted_temperature, name="temperature"),
                                                          tool(counted_humidity, name="humidity")],
                                            maximum_iteration_count=1)
        await sequential.workflow.run(sequential.prompt)
        self.assertEqual(sequential.manager.observation_count, 1)
        self.assertEqual(sequential_requests[1][1]["tool_choice"], "required")
        self.assertEqual([entry["function"]["name"] for entry in sequential_requests[1][1]["tools"]], ["humidity"])
        self.assertFalse(sequential_requests[2][1].get("tools"))
        self.assertEqual(counts, {"temperature": 3, "humidity": 3})

    async def test_azure_v1_and_legacy_transports_use_shared_factories(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                sdk, requests = self.sdk(["Proposal", "APPROVED: okay", "EXECUTED: done"], provider="azure", legacy=legacy)
                sample = gc.build_execution(sdk, provider="azure", motor_tools=[])
                await sample.workflow.run(sample.prompt)
                self.assertEqual(requests[0][1]["model"], "offline-deployment")
                path = requests[0][0].url.path
                self.assertEqual(path, "/openai/deployments/offline-deployment/chat/completions" if legacy
                                 else "/openai/v1/chat/completions")
                if legacy:
                    self.assertEqual(requests[0][0].url.params["api-version"], "2025-04-01-preview")
                else:
                    self.assertEqual(requests[0][0].headers["authorization"], "Bearer offline")

    async def test_structured_review_output_override_does_not_reach_motor_tools(self):
        sdk, requests = self.sdk(["Proposal", '{"decision":"APPROVED","reason":"okay"}', "EXECUTED: done"])
        sample = gc.build_execution(sdk, response_format=gc.ExecutionVerdict, motor_tools=[])
        await sample.workflow.run(sample.prompt)
        self.assertNotIn("response_format", requests[0][1])
        self.assertEqual(requests[1][1]["response_format"]["type"], "json_schema")
        self.assertNotIn("response_format", requests[2][1])
        self.assertEqual(sample.last_response, "EXECUTED: done")

    async def test_all_entrypoint_defaults_provider_and_resource_cleanup(self):
        for project in PROJECTS:
            module = load_sample(project)
            self.assertTrue(inspect.iscoroutinefunction(module.main))
            self.assertEqual(inspect.signature(module.main).parameters["provider"].default, "openai")
            with patch.object(module, "run_example", new=AsyncMock(return_value="done")) as run:
                self.assertEqual(await module.main(), "done")
                self.assertEqual(run.call_args.args[0], project)
                self.assertEqual(run.call_args.kwargs["provider"], "openai")
            sdk = Mock(close=AsyncMock())
            with patch.object(gc, "create_sdk_client", return_value=sdk), patch.object(gc, "build_execution") as execution, \
                 patch.object(gc, "build_investigation") as investigation, patch.object(gc, "run_streaming", new=AsyncMock(return_value="done")):
                self.assertEqual(await gc.run_example(project, provider="azure"), "done")
                builder = investigation if project.endswith("CustomManager") else execution
                self.assertEqual(builder.call_args.kwargs["provider"], "azure")
                sdk.close.assert_awaited_once()
        self.assertEqual(gc.provider_from_cli([]), "openai")
        self.assertEqual(gc.provider_from_cli(["--provider", "azure"]), "azure")
        with self.assertRaises(ValueError):
            await gc.run_example("NotAProject")

    async def test_sdk_cleanup_on_build_failure_run_failure_and_cancellation(self):
        import asyncio
        for error in (RuntimeError("build failed"), RuntimeError("run failed"), asyncio.CancelledError()):
            sdk = Mock(close=AsyncMock())
            with patch.object(gc, "create_sdk_client", return_value=sdk), patch.object(gc, "build_execution") as build, \
                 patch.object(gc, "run_streaming", new=AsyncMock()) as run:
                if str(error) == "build failed":
                    build.side_effect = error
                else:
                    run.side_effect = error
                with self.assertRaises(type(error)):
                    await gc.run_example("GroupChatOrchestration")
                sdk.close.assert_awaited_once()

    async def test_workflow_participant_failure_propagates_without_continuing(self):
        motors = ScriptedAgent("MotorsAgent", [RuntimeError("motor model failed")])
        navigator = ScriptedAgent("NavigatorAgent", [])
        sample = self.execution_workflow(motors, navigator)
        with self.assertRaisesRegex(RuntimeError, "motor model failed"):
            await gc.run_streaming(sample)
        self.assertEqual(len(motors.inputs), 1)
        self.assertEqual(len(navigator.inputs), 0)
        self.assertFalse(sample.workflow._is_running)

    async def test_manager_checkpoint_restores_integer_state_raw_evidence_and_history(self):
        sample = self.investigation_workflow(ScriptedAgent("EnvironmentAgent", []), ScriptedAgent("SafetyAgent", []))
        manager = sample.manager
        manager.observation_count = 2
        manager.observations = [(20, 10), (30, 40)]
        manager._seen_sensor_calls = {"t-1", "h-1", "t-2", "h-2"}
        manager.calibrated = True
        manager.last_agent = "SafetyAgent"
        manager._round_index = 6
        manager.raw_history = self.maintenance_result().messages
        manager._full_conversation = [message("SafetyAgent", "APPROVED: calibrated")]
        state = json.loads(json.dumps(await manager.on_checkpoint_save()))
        restored = self.investigation_workflow(ScriptedAgent("EnvironmentAgent", []), ScriptedAgent("SafetyAgent", [])).manager
        await restored.on_checkpoint_restore(state)
        self.assertEqual((restored.observation_count, restored.iteration_count), (2, 6))
        self.assertIs(type(restored.observation_count), int)
        self.assertTrue(restored.calibrated)
        self.assertEqual(restored.last_agent, "SafetyAgent")
        self.assertEqual(restored.observations, [(20, 10), (30, 40)])
        self.assertEqual(restored._seen_sensor_calls, {"t-1", "h-1", "t-2", "h-2"})
        self.assertEqual(restored.history[0].text, "APPROVED: calibrated")
        self.assertEqual(restored.raw_history[0].contents[0].result, "All sensors have been calibrated.")
        self.assertTrue(restored.should_terminate(restored.history))

    async def test_motor_adapter_consumes_approval_and_denied_feedback_cannot_grant_tools(self):
        inner = Mock(id="motors", name="MotorsAgent", description="Motors")
        inner.run = Mock()
        adapter = gc.GatedMotorAgent(inner, approver_name="HumanAgent", motor_tools=["motor"])
        approval = message("HumanAgent", "HUMAN APPROVED: do it")
        adapter.run([approval])
        self.assertEqual(inner.run.call_args.kwargs["tools"], ["motor"])
        self.assertEqual(inner.run.call_args.kwargs["options"], {"response_format": None, "tool_choice": "auto"})
        adapter.run([approval])
        self.assertEqual(inner.run.call_args.kwargs["tools"], [])
        adapter.run([message("HumanAgent", "HUMAN DENIED: I have NOT APPROVED the path"),
                     message("NavigatorAgent", "APPROVED: advisory")])
        self.assertEqual(inner.run.call_args.kwargs["tools"], [])
        self.assertEqual(inner.run.call_args.kwargs["options"]["tool_choice"], "none")

    async def test_unapproved_model_tool_attempt_has_no_callable_motor_tools(self):
        sdk, requests = self.sdk([("", [("unsafe", "stop", {})]), "Proposal only; no execution"])
        sample = gc.build_execution(sdk, maximum_iteration_count=1)
        await sample.workflow.run(sample.prompt)
        self.assertEqual(sample.manager.iteration_count, 1)
        self.assertEqual(self.stdout.getvalue().count("MOTORS:"), 0)
        self.assertFalse(requests[0][1].get("tools"))
        self.assertEqual(len(requests), 1)
        self.assertIsNone(gc.parse_decision(sample.last_response))

    async def test_actual_calibration_tool_failure_does_not_mark_completed(self):
        async def failing_calibration() -> str:
            raise RuntimeError("offline calibration failed")
        async def temperature() -> int:
            return 20
        async def humidity() -> int:
            return 20
        sdk, requests = self.sdk([("", [("t1", "temperature", {}), ("h1", "humidity", {})]),
                                  "Observation: temperature = 20 Celsius; humidity = 20%.", "CALIBRATION: second",
                                  ("", [("t2", "temperature", {}), ("h2", "humidity", {})]),
                                  "Observation: temperature = 20 Celsius; humidity = 20%.", "CALIBRATION: calibrate",
                                  ("", [("failed", "failing_calibration", {})]), "Calibration failed.",
                                  "APPROVED: unverified model claim"])
        sample = gc.build_investigation(sdk, sensor_tools=[tool(temperature), tool(humidity)],
                                        calibration_tool=tool(failing_calibration), maximum_iteration_count=6)
        await sample.workflow.run(sample.prompt)
        self.assertFalse(sample.manager.calibrated)
        self.assertEqual(sample.manager.iteration_count, 6)
        self.assertEqual(sample.manager.observation_count, 2)
        self.assertIn("Error: Function failed.", str(requests[7][1]["messages"]))
        self.assertIn("Calibration completed: False", str(requests[-1][1]["messages"]))
        self.assertTrue(any(c.exception for m in sample.manager.raw_history for c in m.contents
                            if c.type == "function_result"))

    async def test_mock_sensor_ranges_are_source_integers(self):
        with patch.object(gc.random, "randrange", side_effect=[-20, 99]) as random, \
             patch.object(gc.asyncio, "sleep", new=AsyncMock()) as delay:
            self.assertEqual(await gc.read_temperature(), -20)
            self.assertEqual(await gc.read_humidity(), 99)
            self.assertEqual(random.call_args_list[0].args, (-20, 100))
            self.assertEqual(random.call_args_list[1].args, (0, 100))
            self.assertEqual(await gc.calibrate_sensors(), "All sensors have been calibrated.")
            self.assertEqual(delay.await_count, 3)
            self.assertTrue(all(call.args == (0.5,) for call in delay.call_args_list))


if __name__ == "__main__":
    unittest.main()
