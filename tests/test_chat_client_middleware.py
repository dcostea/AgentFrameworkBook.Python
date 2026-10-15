"""Offline regression tests for chat-client policy and pipeline placement."""

import asyncio
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import AsyncOpenAI

with patch("dotenv.load_dotenv", return_value=False):
    from agent_framework import Agent, ChatContext, ChatResponse, Content, Message
    from agent_framework.exceptions import ChatClientException
    from shared import chat_client_middleware_examples as policies, motor_tools

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "MiddlewareBaseline": "baseline", "ChatClientSharedFunctionMiddleware": "shared",
    "ChatClientResponseMiddleware": "response", "ChatClientFunctionCallingMiddleware": "function",
    "MiddlewareMixed": "mixed",
}


def completion(text="Done.", *, calls=None, tokens=12, response_id="chat-policy-test"):
    return {
        "id": response_id, "object": "chat.completion", "created": 1, "model": "test-model",
        "usage": {"prompt_tokens": tokens - 2, "completion_tokens": 2, "total_tokens": tokens},
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": {
            "role": "assistant", "content": None if calls else text,
            **({"tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                               for cid, name, args in calls]} if calls else {}),
        }}],
    }


def stream_payload(text="Streamed.", tokens=12, calls=None):
    base = {"id": "stream-policy-test", "object": "chat.completion.chunk", "created": 1, "model": "test-model"}
    deltas = [{"role": "assistant"}]
    if calls:
        deltas.append({"tool_calls": [{"index": index, "id": cid, "type": "function",
                                      "function": {"name": name, "arguments": json.dumps(args)}}
                                     for index, (cid, name, args) in enumerate(calls)]})
    else:
        deltas.extend({"content": part} for part in (text[:3], "", text[3:]))
    chunks = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]})
    chunks.append({**base, "choices": [], "usage": {"prompt_tokens": tokens-2, "completion_tokens": 2, "total_tokens": tokens}})
    data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=data.encode(), headers={"content-type": "text/event-stream"})


class ChatMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "OPENAI_CHAT_MODEL_ID": "test-model"}, clear=True))
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.object(motor_tools, "_DELAY_SECONDS", 0))
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(patch.object(httpx.AsyncHTTPTransport, "handle_async_request",
                                      new=AsyncMock(side_effect=AssertionError("Real network forbidden"))))

    def sdk(self, responses):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            if len(requests) > len(responses):
                raise AssertionError("Unexpected provider call")
            result = responses[len(requests)-1]
            return result if isinstance(result, httpx.Response) else httpx.Response(200, json=result)

        sdk = AsyncOpenAI(api_key="offline", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        self.addAsyncCleanup(sdk.close)
        return sdk, requests

    def sample(self, name):
        spec = importlib.util.spec_from_file_location("policy_"+name, ROOT / "09.ChatClientMiddleware" / name / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    async def test_all_entrypoints_preserve_prompts_lifecycle_and_limit(self):
        for name, kind in SAMPLES.items():
            with self.subTest(sample=name):
                sdk, requests = self.sdk([completion() for _ in policies.PROMPTS[kind]])
                sample = self.sample(name)
                with patch.object(sample, "create_sdk_client", return_value=sdk):
                    await sample.main()
                expected = min(len(policies.PROMPTS[kind]), 2) if kind in {"shared", "mixed"} else len(policies.PROMPTS[kind])
                self.assertEqual(len(requests), expected)
                self.assertTrue(sdk.is_closed())
                if kind in {"shared", "mixed"}:
                    self.assertIn("EXCEPTION: Request limit exceeded", self.output.getvalue())
        with self.assertRaises(ValueError):
            policies.build_demo(None, "not-a-demo")

    async def test_email_redaction_transient_and_preserves_multimodal_protocol(self):
        original = Message("user", [Content.from_text("A+tag@example.org and other@example.net"),
                                    Content.from_data(b"jpeg", "image/jpeg")], author_name="operator", message_id="u1")
        assistant = Message("assistant", [Content.from_function_call("c1", "stop", arguments={})])
        tool = Message("tool", [Content.from_function_result("c1", result="email@example.com")])
        context = ChatContext(Mock(), [original, assistant, tool], {})
        observed = []

        async def next_handler():
            observed.extend(context.messages)
            context.result = ChatResponse(messages=Message("assistant", ["ok"]))

        await policies.RemoveEmail().process(context, next_handler)
        self.assertEqual(observed[0].text, "[REDACTED-EMAIL] and [REDACTED-EMAIL]")
        self.assertEqual(original.text, "A+tag@example.org and other@example.net")
        self.assertEqual(observed[0].message_id, "u1")
        self.assertEqual(observed[0].author_name, "operator")
        self.assertIs(observed[0].contents[1], original.contents[1])
        self.assertIs(observed[1], assistant)
        self.assertIs(observed[2], tool)

    async def test_chat_redaction_keeps_original_session_history_and_reapplies(self):
        sdk, requests = self.sdk([completion(), completion()])
        agent = policies.build_demo(sdk, "shared")
        session = agent.create_session()
        await agent.run("Contact john.doe@example.com", session=session)
        await agent.run("Turn left", session=session)
        self.assertNotIn("john.doe@example.com", json.dumps(requests))
        self.assertIn("[REDACTED-EMAIL]", json.dumps(requests))
        self.assertIn("john.doe@example.com", json.dumps(session.to_dict()))
        self.assertEqual(agent.client.chat_middleware[0].request_count, 2)

    async def test_request_limit_shared_client_counts_denials_and_blocks_network(self):
        sdk, requests = self.sdk([completion(), completion()])
        first = policies.build_demo(sdk, "shared")
        second = Agent(client=first.client, instructions="Second agent")
        await first.run("one")
        await second.run("two")
        for count in (3, 4):
            with self.assertRaises(policies.LimitExceededException) as caught:
                await first.run("blocked")
            self.assertEqual(caught.exception.current_count, count)
            self.assertEqual(caught.exception.max_requests, 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(policies.LimitRequests().request_count, 0)
        with self.assertRaises(ValueError):
            policies.LimitRequests(0)

    async def test_token_budget_skips_timestamp_and_keeps_synthetic_response(self):
        sdk, requests = self.sdk([completion(tokens=15)])
        agent = policies.build_demo(sdk, "response")
        budget = agent.client.chat_middleware[0]
        budget.max_tokens = 15
        first = await agent.run("one")
        second = await agent.run("two")
        self.assertTrue(first.text.startswith("["))
        self.assertEqual(second.text, "Token budget exhausted.")
        self.assertEqual(len(requests), 1)
        self.assertEqual(budget.tokens_count, 15)
        self.assertEqual(self.output.getvalue().count("Stamping response"), 1)
        with self.assertRaises(ValueError):
            policies.EnforceTokenBudget(0)

    async def test_usage_fallback_overshoot_and_missing_usage(self):
        budget = policies.EnforceTokenBudget(20)
        for usage, expected in (({"input_token_count": 10, "output_token_count": 7}, 17),
                                ({"total_token_count": 9, "input_token_count": 90}, 26), (None, 26)):
            result = ChatResponse(messages=[Message("assistant", ["response"])], usage_details=usage)
            self.assertIs(budget.record_usage(result), result)
            self.assertEqual(budget.tokens_count, expected)

    async def test_timestamp_retains_tool_content_and_metadata(self):
        call = Content.from_function_call("c1", "stop", arguments={})
        response = ChatResponse(messages=[Message("assistant", ["First", call, Content.from_text("second")]),
                                          Message("tool", [Content.from_function_result("c1", result="ok")])],
                                response_id="rid", model="model", finish_reason="tool_calls", usage_details={"total_token_count": 9})
        context = ChatContext(Mock(), [], {})

        async def handler():
            context.result = response

        await policies.AddTimestamp(clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)).process(context, handler)
        self.assertEqual(response.messages[0].text, "[2026-01-01T00:00:00+00:00] First second")
        self.assertIs(response.messages[0].contents[1], call)
        self.assertEqual(response.messages[1].contents[0].result, "ok")
        self.assertEqual((response.response_id, response.model, response.finish_reason), ("rid", "model", "tool_calls"))
        self.assertEqual(response.usage_details, {"total_token_count": 9})

    async def test_backward_cap_does_not_limit_forward_and_invokes_once(self):
        calls = [("f", "forward", {"distance": 10}), ("b", "backward", {"distance": 8}),
                 ("exact", "backward", {"distance": 5})]
        sdk, requests = self.sdk([completion(calls=calls), completion()])
        response = await policies.build_demo(sdk, "function").run("move")
        self.assertEqual(response.text, "Done.")
        results = {m["tool_call_id"]: m["content"] for m in requests[1]["messages"] if m["role"] == "tool"}
        self.assertIn("10 meters", results["f"])
        self.assertIn("5 meters", results["b"])
        self.assertIn("5 meters", results["exact"])
        self.assertEqual(self.output.getvalue().count("MOTORS:"), 3)
        self.assertEqual(self.output.getvalue().count("[Audit] Function call"), 3)
        self.assertEqual(self.output.getvalue().count("constrained from"), 1)

    async def test_baseline_leaves_email_and_backward_unrestricted(self):
        sdk, requests = self.sdk([completion(calls=[("b", "backward", {"distance": 10})]), completion()])
        response = await policies.build_demo(sdk, "baseline").run("Email email@example.com; reverse ten meters")
        self.assertEqual(response.text, "Done.")
        self.assertIn("email@example.com", json.dumps(requests[0]))
        self.assertIn("10 meters", next(m["content"] for m in requests[1]["messages"] if m["role"] == "tool"))
        self.assertNotIn("[ChatClient]", self.output.getvalue())

    async def test_standard_roundtrip_vs_mixed_external_layer_placement(self):
        for kind, expected_count in (("response", 2), ("mixed", 1)):
            with self.subTest(kind=kind):
                before = self.output.getvalue().count("Stamping response")
                sdk, requests = self.sdk([completion(calls=[("b", "backward", {"distance": 8})], tokens=12), completion(tokens=17)])
                agent = policies.build_demo(sdk, kind)
                response = await agent.run("Move; email me at user@example.com")
                self.assertEqual(len(requests), 2)
                self.assertEqual(self.output.getvalue().count("Stamping response")-before, expected_count)
                if kind == "mixed":
                    self.assertEqual(agent.client.chat_middleware[0].request_count, 1)
                    self.assertEqual(agent.client.chat_middleware[2].tokens_count, 29)
                    self.assertNotIn("user@example.com", json.dumps(requests))
                    self.assertIn("5 meters", next(m["content"] for m in requests[1]["messages"] if m["role"] == "tool"))
                else:
                    self.assertEqual(agent.client.chat_middleware[0].tokens_count, 29)
                self.assertEqual(response.text.count("] Done."), 1)

    async def test_mixed_order_denial_skips_all_inner_work(self):
        sdk, requests = self.sdk([completion(), completion()])
        agent = policies.build_demo(sdk, "mixed")
        await agent.run("Contact user@example.com")
        await agent.run("second")
        lines = self.output.getvalue()
        self.assertLess(lines.index("[Limit] PRE"), lines.index("[Email] PRE"))
        self.assertLess(lines.index("[Timestamp]"), lines.index("[Email] POST"))
        before = self.output.getvalue()
        with self.assertRaises(policies.LimitExceededException):
            await agent.run("third")
        new = self.output.getvalue()[len(before):]
        self.assertIn("Request 3", new)
        self.assertNotIn("[Email]", new)
        self.assertNotIn("[Timestamp]", new)
        self.assertEqual(len(requests), 2)

    async def test_streaming_timestamp_aggregation_budget_and_final_metadata(self):
        sdk, requests = self.sdk([stream_payload(), stream_payload("Not reached")])
        agent = policies.build_demo(sdk, "response")
        budget = agent.client.chat_middleware[0]
        budget.max_tokens = 12
        stream = agent.run("hello", stream=True)
        pieces = [update.text async for update in stream if update.text]
        final = await stream.get_final_response()
        self.assertEqual("".join(pieces), final.text)
        self.assertEqual(final.text.count("["), 1)
        self.assertTrue(final.text.endswith("Streamed."))
        self.assertEqual(final.response_id, "stream-policy-test")
        self.assertEqual(final.usage_details["total_token_count"], 12)
        self.assertEqual(budget.tokens_count, 12)
        stream = agent.run("over budget", stream=True)
        text = "".join([update.text async for update in stream])
        self.assertEqual(text, "Token budget exhausted.")
        self.assertEqual((await stream.get_final_response()).text, text)
        self.assertEqual(len(requests), 1)

    async def test_mixed_streaming_tool_loop_counts_once_and_returns_results(self):
        sdk, requests = self.sdk([stream_payload(calls=[("b", "backward", {"distance": 9})], tokens=12),
                                  stream_payload("Moved.", tokens=17)])
        agent = policies.build_demo(sdk, "mixed")
        stream = agent.run("Reverse", stream=True)
        parts = [u.text async for u in stream if u.text]
        final = await stream.get_final_response()
        self.assertEqual("".join(parts), final.text)
        self.assertTrue(final.text.endswith("Moved."))
        self.assertEqual(final.text.count("["), 1)
        self.assertEqual(agent.client.chat_middleware[0].request_count, 1)
        self.assertEqual(agent.client.chat_middleware[2].tokens_count, 29)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.output.getvalue().count("MOTORS:"), 1)
        self.assertIn("5 meters", next(m["content"] for m in requests[1]["messages"] if m["role"] == "tool"))

    async def test_stream_error_cleanup_and_provider_failure_propagation(self):
        class BrokenStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                chunk = {"id": "broken", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "start"}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                raise RuntimeError("stream broke")

        broken = httpx.Response(200, stream=BrokenStream(), headers={"content-type": "text/event-stream"})
        sdk, requests = self.sdk([broken])
        agent = policies.build_demo(sdk, "shared")
        stream = agent.run("hello", stream=True)
        with self.assertRaisesRegex(ChatClientException, "stream broke"):
            async for _ in stream:
                pass
        self.assertIn("[Email] POST: Completed", self.output.getvalue())
        self.assertIn("[Limit] POST: Request 1 completed", self.output.getvalue())
        context = ChatContext(Mock(), [Message("user", ["hello"])], {})
        failure = AsyncMock(side_effect=RuntimeError("model failed"))
        for policy in (policies.LimitRequests(), policies.RemoveEmail(), policies.EnforceTokenBudget(), policies.AddTimestamp()):
            with self.subTest(policy=type(policy).__name__), self.assertRaisesRegex(RuntimeError, "model failed"):
                await policy.process(context, failure)
        self.assertEqual(failure.await_count, 4)

    async def test_streaming_entrypoint_for_azure_routes_explicit_sdk_provider(self):
        sample = self.sample("ChatClientResponseMiddleware")
        sdk, requests = self.sdk([stream_payload(), stream_payload()])
        with patch.dict(os.environ, {"AZURE_OPENAI_DEPLOYMENT_NAME": "azure-model"}), \
             patch.object(sample, "create_sdk_client", return_value=sdk) as create:
            await sample.main("azure", True)
        create.assert_called_once_with("azure")
        self.assertEqual([r["model"] for r in requests], ["azure-model", "azure-model"])
        self.assertTrue(sdk.is_closed())


if __name__ == "__main__":
    unittest.main()
