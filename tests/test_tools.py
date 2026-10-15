"""Offline tool-mode, manual-invocation, and human-approval regression tests."""

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
    from agent_framework import AgentResponse, Content, Message
    from shared import motor_tools, tool_examples

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("AgentWithTools", "AgentWithToolMode", "AgentWithMultipleToolCalls",
           "AgentWithManualToolInvocation", "AgentWithFunctionApproval")


def load_sample(name):
    spec = importlib.util.spec_from_file_location(f"tools_{name}", ROOT / "06.Tools" / name / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(text="Done.", calls=None):
    return {
        "id": "chat-tools-test", "object": "chat.completion", "created": 1, "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": {
            "role": "assistant", "content": None if calls else text,
            **({"tool_calls": [{"id": cid, "type": "function", "function": {
                "name": name, "arguments": json.dumps(args),
            }} for cid, name, args in calls]} if calls else {}),
        }}],
    }


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "OPENAI_CHAT_MODEL_ID": "test-model"}, clear=True))
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.object(motor_tools, "_DELAY_SECONDS", 0))
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request",
                                      new=AsyncMock(side_effect=AssertionError("Live network forbidden"))))

    def sdk(self, responses):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            index = len(requests) - 1
            if index >= len(responses):
                raise AssertionError("Unexpected model request")
            return httpx.Response(200, json=responses[index])

        sdk = AsyncOpenAI(api_key="offline", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        self.addAsyncCleanup(sdk.close)
        return sdk, requests

    async def test_all_five_scripts_import_without_requests(self):
        for name in SAMPLES:
            with self.subTest(sample=name):
                self.assertTrue(asyncio.iscoroutinefunction(load_sample(name).main))

    async def test_automatic_tools_and_matching_results(self):
        calls = [("right-1", "turn_right", {"angle": 90}), ("forward-1", "forward", {"distance": 3}),
                 ("left-1", "turn_left", {"angle": 90}), ("back-1", "backward", {"distance": 2}),
                 ("stop-1", "stop", {})]
        sdk, requests = self.sdk([payload(calls=calls), payload()])
        response = await tool_examples.build_agent(sdk).run("Avoid the tree")
        self.assertEqual(response.text, "Done.")
        results = [m for m in requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual({m["tool_call_id"] for m in results}, {c[0] for c in calls})
        self.assertTrue(any("3 meters" in m["content"] for m in results))
        self.assertEqual(self.output.getvalue().count("MOTORS:"), 5)
        tool_examples.print_tools(response.messages)
        self.assertIn("TOOL CALL right-1: turn_right", self.output.getvalue())
        self.assertIn("TOOL RESP stop-1: stopped.", self.output.getvalue())

    async def test_manual_loop_does_not_auto_execute_and_keeps_call_ids(self):
        sdk, requests = self.sdk([payload(calls=[("c1", "turn_left", {"angle": 45}), ("c2", "stop", {})]), payload()])
        tools = tool_examples.create_motor_tools()
        agent = tool_examples.build_agent(sdk, tools=tools, manual=True,
                                          options={"allow_multiple_tool_calls": True, "tool_choice": "auto"})
        self.assertFalse(agent.client.function_invocation_configuration["enabled"])
        response = await tool_examples.run_manual_tools(agent, "Go left then stop", tools)
        self.assertEqual(response.text, "Done.")
        self.assertEqual(self.output.getvalue().count("MOTORS:"), 2)
        self.assertTrue(requests[0]["parallel_tool_calls"])
        self.assertEqual([m["tool_call_id"] for m in requests[1]["messages"] if m["role"] == "tool"], ["c1", "c2"])
        self.assertEqual(len([m for m in requests[1]["messages"] if m["role"] == "user"]), 1)

    async def test_manual_loops_multiple_rounds_and_plain_answer(self):
        sdk, requests = self.sdk([payload(calls=[("first", "forward", {"distance": 4})]),
                                  payload(calls=[("last", "stop", {})]), payload()])
        tools = tool_examples.create_motor_tools()
        response = await tool_examples.run_manual_tools(tool_examples.build_agent(sdk, tools=tools, manual=True),
                                                       "move", tools)
        self.assertEqual(response.text, "Done.")
        self.assertEqual(len(requests), 3)
        self.assertEqual([m["tool_call_id"] for m in requests[-1]["messages"] if m["role"] == "tool"], ["first", "last"])
        sdk, requests = self.sdk([payload("No moves needed.")])
        response = await tool_examples.run_manual_tools(tool_examples.build_agent(sdk, manual=True), "hello", tools)
        self.assertEqual(response.text, "No moves needed.")
        self.assertEqual(len(requests), 1)

    async def test_manual_rejects_unknown_invalid_arguments_and_round_limit(self):
        for name, args, expected in (("erase_disk", {}, RuntimeError), ("forward", [2], ValueError),
                                     ("forward", {"distance": "not-a-number"}, TypeError)):
            with self.subTest(name=name, args=args):
                sdk, requests = self.sdk([payload(calls=[("bad", name, args)])])
                tools = tool_examples.create_motor_tools()
                with self.assertRaises(expected):
                    await tool_examples.run_manual_tools(tool_examples.build_agent(sdk, manual=True), "test", tools)
                self.assertEqual(len(requests), 1)
        fake = Mock()
        fake.create_session.return_value = object()
        fake.run = AsyncMock(return_value=AgentResponse(messages=[Message("assistant", [
            Content.from_function_call("again", "stop", arguments={}),
        ])]))
        with self.assertRaisesRegex(RuntimeError, "exceeded 2 rounds"):
            await tool_examples.run_manual_tools(fake, "test", tool_examples.create_motor_tools(), max_rounds=2)
        self.assertEqual(fake.run.await_count, 3)
        for function in (tool_examples.run_manual_tools, tool_examples.run_with_approval):
            with self.assertRaises(ValueError):
                if function is tool_examples.run_manual_tools:
                    await function(fake, "test", [], max_rounds=0)
                else:
                    await function(fake, "test", max_rounds=0)

    async def test_only_stop_and_backward_tools_require_approval(self):
        tools = tool_examples.create_motor_tools(require_approval=True)
        self.assertEqual({t.name for t in tools if t.approval_mode == "always_require"}, {"stop", "backward"})
        self.assertEqual({t.name for t in tool_examples.create_motor_tools() if t.approval_mode != "never_require"}, set())

    async def test_approval_denial_is_not_executed_and_other_batch_tools_execute(self):
        sdk, requests = self.sdk([payload(calls=[("back", "backward", {"distance": 4}), ("fwd", "forward", {"distance": 2}),
                                               ("halt", "stop", {})]), payload("Denied backward; moved and stopped.")])
        agent = tool_examples.build_agent(sdk, tools=tool_examples.create_motor_tools(require_approval=True))
        prompts = []
        answers = iter(("n", "Y"))

        def read_input(prompt):
            prompts.append(prompt)
            self.assertNotIn("MOTORS:", self.output.getvalue())
            return next(answers)

        response = await tool_examples.run_with_approval(agent, "danger", read_input=read_input)
        self.assertEqual(response.text, "Denied backward; moved and stopped.")
        self.assertEqual(len(prompts), 2)
        self.assertIn("backward", prompts[0])
        self.assertIn("stop", prompts[1])
        self.assertNotIn("MOTORS: Backward", self.output.getvalue())
        self.assertIn("MOTORS: Forward", self.output.getvalue())
        self.assertIn("MOTORS: Stop", self.output.getvalue())
        results = {m["tool_call_id"]: m["content"] for m in requests[1]["messages"] if m["role"] == "tool"}
        self.assertEqual(set(results), {"back", "fwd", "halt"})
        self.assertIn("reject", results["back"].lower())
        self.assertIn("2 meters", results["fwd"])

    async def test_approval_decisions_blank_yes_no_and_eof(self):
        request = Content.from_function_approval_request("approval", Content.from_function_call("approval", "stop", arguments={}))
        for answer, expected in (("", True), (" Y ", True), ("yes", True), ("n", False), ("unknown", False)):
            with self.subTest(answer=answer):
                self.assertEqual(tool_examples.prompt_for_approval(request, lambda _: answer), expected)
        self.assertFalse(tool_examples.prompt_for_approval(request, Mock(side_effect=EOFError)))
        fake = Mock()
        fake.create_session.return_value = object()
        fake.run = AsyncMock(return_value=AgentResponse(messages=[Message("assistant", [request])]))
        with self.assertRaisesRegex(RuntimeError, "exceeded 2 rounds"):
            await tool_examples.run_with_approval(fake, "test", read_input=lambda _: "n", max_rounds=2)
        self.assertEqual(fake.run.await_count, 3)

    async def test_tool_modes_translate_and_required_resets_after_first_call(self):
        sample = load_sample("AgentWithToolMode")
        for mode in ("auto", "none", "required", "backward"):
            with self.subTest(mode=mode):
                responses = [payload()] if mode in {"auto", "none"} else [payload(calls=[("b", "backward", {"distance": 1})]), payload()]
                sdk, requests = self.sdk(responses)
                agent = tool_examples.build_agent(sdk, options={"tool_choice": sample.tool_choice(mode)})
                response = await agent.run("command")
                self.assertEqual(response.text, "Done.")
                if mode == "backward":
                    self.assertEqual(requests[0]["tool_choice"], {"type": "function", "function": {"name": "backward"}})
                else:
                    self.assertEqual(requests[0]["tool_choice"], mode)
                if mode in {"required", "backward"}:
                    self.assertNotIn("tool_choice", requests[1])
        with self.assertRaises(ValueError):
            sample.tool_choice("invalid")

    async def test_parallel_option_and_entrypoint_sdk_cleanup(self):
        sample = load_sample("AgentWithMultipleToolCalls")
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                sdk, requests = self.sdk([payload()])
                with patch.object(sample, "create_sdk_client", return_value=sdk):
                    await sample.main(parallel)
                self.assertEqual(requests[0]["parallel_tool_calls"], parallel)
                self.assertTrue(sdk.is_closed())
        sample = load_sample("AgentWithManualToolInvocation")
        sdk, requests = self.sdk([payload(calls=[("halt", "stop", {})]), payload()])
        with patch.object(sample, "create_sdk_client", return_value=sdk):
            await sample.main()
        self.assertEqual(len(requests), 2)
        self.assertTrue(sdk.is_closed())

    async def test_automatic_and_tool_mode_entrypoints_preserve_stateless_runs(self):
        sample = load_sample("AgentWithTools")
        sdk, requests = self.sdk([payload(calls=[("stop", "stop", {})]), payload()])
        with patch.object(sample, "create_sdk_client", return_value=sdk):
            await sample.main()
        self.assertEqual(len(requests), 2)
        self.assertTrue(sdk.is_closed())
        self.assertIn("Call the matching tools", requests[0]["messages"][0]["content"])
        sample = load_sample("AgentWithToolMode")
        sdk, requests = self.sdk([payload("First answer"), payload("Second answer")])
        with patch.object(sample, "create_sdk_client", return_value=sdk):
            await sample.main("none")
        self.assertEqual([r["tool_choice"] for r in requests], ["none", "none"])
        self.assertEqual([len([m for m in r["messages"] if m["role"] == "user"]) for r in requests], [1, 1])
        self.assertIn("What movements can you perform?", requests[1]["messages"][-1]["content"])
        self.assertTrue(sdk.is_closed())

    async def test_function_approval_entrypoint_closes_client(self):
        sample = load_sample("AgentWithFunctionApproval")
        sdk, requests = self.sdk([payload(calls=[("stop", "stop", {})]), payload()])

        async def approve(agent, prompt):
            return await tool_examples.run_with_approval(agent, prompt, read_input=lambda _: "Y")

        with patch.object(sample, "create_sdk_client", return_value=sdk), \
             patch.object(sample, "run_with_approval", side_effect=approve):
            await sample.main()
        self.assertEqual(len(requests), 2)
        self.assertIn("Danger ahead", requests[0]["messages"][-1]["content"])
        self.assertIn("AI Tool 'stop': Approved", self.output.getvalue())
        self.assertTrue(sdk.is_closed())


if __name__ == "__main__":
    unittest.main()
