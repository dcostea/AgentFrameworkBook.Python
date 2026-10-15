"""Offline chapter 05 regression tests using the real framework and SDK adapters."""

import base64
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, InternalServerError

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "05.Agents"
ENV = {
    "OPENAI_API_KEY": "offline-openai-key",
    "OPENAI_CHAT_MODEL_ID": "offline-chat-model",
    "OPENAI_RESPONSES_MODEL_ID": "offline-responses-model",
    "AZURE_OPENAI_ENDPOINT": "https://azure.invalid",
    "AZURE_OPENAI_API_KEY": "offline-azure-key",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "offline-azure-chat",
    "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME": "offline-azure-responses",
    "AZURE_OPENAI_AUDIO_ENDPOINT": "https://audio.invalid",
    "AZURE_OPENAI_AUDIO_API_KEY": "offline-audio-key",
    "AZURE_OPENAI_AUDIO_DEPLOYMENT_NAME": "offline-audio-model",
}
PROJECTS = (
    "Agent", "AgentWithBackgroundResponses", "AgentWithChatMessages",
    "AgentWithChatMessagesAndMultiModal", "AgentWithInServiceConversation",
    "AgentWithSerialization", "AgentWithSession", "AgentWithStreaming",
    "AgentWithStructuredOutput", "MultipleAgentsWithSameSession",
)
MOVES = '[{"move":"turn right","angle":90},{"move":"forward","distance":5}]'
STRUCTURED = '{"Steps":[{"Move":"stop","Value":null},{"Move":"forward","Value":"5"}]}'

with patch.dict(os.environ, ENV, clear=True), patch("dotenv.load_dotenv", return_value=False):
    sys.path.insert(0, str(ROOT))
    from agent_framework import Agent, AgentSession, InMemoryHistoryProvider, Message
    from agent_framework.exceptions import ChatClientException
    from pydantic import ValidationError
    from shared import agent_examples, chat_clients


def load_sample(name):
    spec = importlib.util.spec_from_file_location(f"offline_05_{name}", CHAPTER / name / "main.py")
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def completion_payload(text=MOVES):
    return {
        "id": "chatcmpl_offline", "object": "chat.completion", "created": 1760000000,
        "model": "served-offline",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "refusal": None},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 23, "completion_tokens": 7, "total_tokens": 30},
    }


def responses_payload(status="completed", *, response_id="resp_offline", text=MOVES, conversation_id=None):
    payload = {
        "id": response_id, "object": "response", "created_at": 1760000000,
        "model": "served-offline", "status": status, "background": True, "store": True,
        "error": {"code": "server_error", "message": "Offline failure"} if status == "failed" else None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": None, "max_output_tokens": 512, "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [], "temperature": 1.0, "top_p": 1.0,
        "output": [{"id": "msg_offline", "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}]
                  if status == "completed" else [],
        "usage": {"input_tokens": 23, "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 30}
                 if status == "completed" else None,
    }
    if conversation_id:
        payload["conversation"] = {"id": conversation_id}
    return payload


def streaming_payload(text=MOVES):
    chunks = []
    for index, part in enumerate((text[:7], "", text[7:])):
        chunks.append({
            "id": "chatcmpl_stream", "object": "chat.completion.chunk", "created": 1760000000,
            "model": "served-offline", "choices": [{"index": 0, "delta": {
                "content": part, **({"role": "assistant"} if index == 0 else {}),
            }, "finish_reason": None}],
        })
    chunks.append({
        "id": "chatcmpl_stream", "object": "chat.completion.chunk", "created": 1760000000,
        "model": "served-offline", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    chunks.append({
        "id": "chatcmpl_stream", "object": "chat.completion.chunk", "created": 1760000000,
        "model": "served-offline", "choices": [],
        "usage": {"prompt_tokens": 23, "completion_tokens": 7, "total_tokens": 30},
    })
    data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=data.encode("utf-8"), headers={"Content-Type": "text/event-stream"})


def wire_text(message):
    contents = message.get("content", "")
    if isinstance(contents, str):
        return contents
    return "".join(item.get("text", "") for item in contents or [])


def service_item(role, text, item_id):
    return {"type": "message", "id": item_id, "role": role, "status": "completed",
            "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text,
                         "annotations": []}]}


class OfflineAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, ENV, clear=True))
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch.object(
            httpx.AsyncHTTPTransport, "handle_async_request",
            new=AsyncMock(side_effect=AssertionError("Real network requests are forbidden")),
        ))
        self.enterContext(patch.object(
            httpx.HTTPTransport, "handle_request", side_effect=AssertionError("Real network requests are forbidden"),
        ))
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.temp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def configured_sdk(self, responder=None, *, provider="openai", audio=False):
        requests = []

        def handle(request):
            record = {
                "method": request.method, "path": request.url.path, "host": request.url.host,
                "query": dict(request.url.params),
                "body": json.loads(request.content) if request.content else None,
            }
            requests.append(record)
            payload = (responder or (lambda record, index: completion_payload()))(record, len(requests) - 1)
            return payload if isinstance(payload, httpx.Response) else httpx.Response(200, json=payload)

        def construct(cls, **kwargs):
            sdk = cls(**kwargs, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
            self.addAsyncCleanup(sdk.close)
            return sdk

        with patch.object(chat_clients, "AsyncOpenAI", side_effect=lambda **kw: construct(AsyncOpenAI, **kw)), \
             patch.object(chat_clients, "AsyncAzureOpenAI", side_effect=lambda **kw: construct(AsyncAzureOpenAI, **kw)):
            sdk = chat_clients.create_sdk_client(provider, audio=audio)
        return sdk, requests

    def chat_agent(self, sdk, *, instructions=agent_examples.ROBOT_INSTRUCTIONS, name="RobotCarAgent", history=False):
        return Agent(
            client=chat_clients.create_completion_client(sdk), instructions=instructions, name=name,
            context_providers=[InMemoryHistoryProvider()] if history else None,
        )

    def responses_agent(self, sdk, *, instructions=agent_examples.ROBOT_INSTRUCTIONS):
        return Agent(client=chat_clients.create_responses_client(sdk), instructions=instructions)

    async def test_all_ten_import_without_side_effects(self):
        self.assertEqual({p.name for p in CHAPTER.iterdir() if p.is_dir()}, set(PROJECTS))
        with patch.object(chat_clients, "create_sdk_client", side_effect=AssertionError("Import invoked the SDK")):
            for name in PROJECTS:
                with self.subTest(project=name):
                    module = load_sample(name)
                    self.assertTrue(inspect.iscoroutinefunction(module.main))
                    self.assertTrue(inspect.iscoroutinefunction(module.run_example))
                    self.assertEqual(Path(module.__file__).parent.name, name)
                    self.assertIn("parents[2]", Path(module.__file__).read_text(encoding="utf-8"))
        self.assertEqual(self.output.getvalue(), "")

    async def test_stateless_runs_do_not_replay_previous_turn(self):
        sdk, requests = self.configured_sdk(lambda record, index: completion_payload(MOVES if index == 0 else "No history."))
        agent = self.chat_agent(sdk, instructions=agent_examples.JSON_INSTRUCTIONS)
        first, second = await load_sample("Agent").run_example(agent)
        self.assertEqual((first.text, second.text), (MOVES, "No history."))
        self.assertEqual(first.usage_details["total_token_count"], 30)
        self.assertEqual(first.messages[0].author_name, "RobotCarAgent")
        for index, prompt in enumerate((agent_examples.TREE_PROMPT, agent_examples.STATELESS_FOLLOWUP_PROMPT)):
            messages = requests[index]["body"]["messages"]
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            self.assertEqual(wire_text(messages[0]), agent_examples.JSON_INSTRUCTIONS)
            self.assertEqual(wire_text(messages[1]), prompt)
        self.assertEqual(agent.context_providers, [])

    async def test_explicit_chat_history_contains_original_moves_and_appends_reply(self):
        sdk, requests = self.configured_sdk(lambda record, index: completion_payload("forward, distance 5"))
        response, conversation = await load_sample("AgentWithChatMessages").run_example(self.chat_agent(sdk))
        messages = requests[0]["body"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        prior = json.loads(wire_text(messages[2]).removeprefix("```json\n").removesuffix("\n```"))
        self.assertEqual(len(prior), 8)
        self.assertEqual(prior[-2], {"move": "forward", "distance": "5"})
        self.assertEqual(wire_text(messages[-1]), agent_examples.FOLLOWUP_PROMPT)
        self.assertEqual(len(conversation), 4)
        self.assertEqual(conversation[-1].role, "assistant")
        self.assertEqual(conversation[-1].text, response.text)

    async def test_local_session_replays_and_saves_both_turns(self):
        sdk, requests = self.configured_sdk(lambda record, index: completion_payload(MOVES if index == 0 else "turn right"))
        first, second, session = await load_sample("AgentWithSession").run_example(self.chat_agent(sdk))
        second_messages = requests[1]["body"]["messages"]
        self.assertEqual([m["role"] for m in second_messages], ["system", "user", "assistant", "user"])
        self.assertEqual(wire_text(second_messages[2]), first.text)
        self.assertEqual(wire_text(second_messages[-1]), agent_examples.FOLLOWUP_PROMPT)
        self.assertIsNone(session.service_session_id)
        history = session.state["in_memory"]["messages"]
        self.assertEqual([m.role for m in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual(history[-1].text, second.text)

    async def test_new_session_does_not_leak_another_sessions_history(self):
        sdk, requests = self.configured_sdk()
        agent = self.chat_agent(sdk)
        first, second = agent.create_session(), agent.create_session()
        await agent.run("Private first session", session=first)
        await agent.run("Unrelated second session", session=second)
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual([m["role"] for m in requests[1]["body"]["messages"]], ["system", "user"])
        self.assertNotIn("Private first session", json.dumps(requests[1]["body"]))
        self.assertEqual(second.state["in_memory"]["messages"][0].text, "Unrelated second session")

    async def test_serialization_writes_python_utf8_and_restores_history(self):
        module = load_sample("AgentWithSerialization")
        text = "Tournez à droite — 避开树"
        sdk, requests = self.configured_sdk(lambda record, index: completion_payload(text if index == 0 else "forward"))
        path = self.temp / "agent_session.json"
        first, followup, restored = await module.run_example(self.chat_agent(sdk, name="MotorsAgent", history=True), path=path)
        snapshot = json.loads(path.read_bytes().decode("utf-8"))
        self.assertEqual(snapshot["type"], "session")
        self.assertEqual(snapshot["session_id"], restored.session_id)
        self.assertNotIn("stateBag", snapshot)
        self.assertNotIn(ENV["OPENAI_API_KEY"], path.read_text(encoding="utf-8"))
        self.assertIn(text.encode("utf-8"), path.read_bytes())
        before = module.load_session(path)
        self.assertEqual([m.role for m in before.state["in_memory"]["messages"]], ["user", "assistant"])
        self.assertIsInstance(before.state["in_memory"]["messages"][0], Message)
        self.assertEqual(before.state["in_memory"]["messages"][1].author_name, "MotorsAgent")
        self.assertEqual(wire_text(requests[1]["body"]["messages"][2]), first.text)
        self.assertEqual(restored.state["in_memory"]["messages"][-1].text, followup.text)
        fresh_agent = self.chat_agent(sdk, name="MotorsAgent", history=True)
        await fresh_agent.run("Resume after restart", session=before)
        self.assertEqual(wire_text(requests[2]["body"]["messages"][2]), text)
        self.assertEqual(wire_text(requests[2]["body"]["messages"][-1]), "Resume after restart")

    async def test_serialization_preserves_service_id_and_nested_state(self):
        module = load_sample("AgentWithSerialization")
        session = AgentSession(session_id="local-id", service_session_id="conv_offline")
        session.state["preferences"] = {"name": "Zoë", "languages": ["日本語", "English"]}
        path = self.temp / "session.json"
        module.save_session(session, path)
        restored = module.load_session(path)
        self.assertEqual(restored.session_id, "local-id")
        self.assertEqual(restored.service_session_id, "conv_offline")
        self.assertEqual(restored.state["preferences"], {"name": "Zoë", "languages": ["日本語", "English"]})
        restored.state["preferences"]["languages"].append("French")
        self.assertEqual(len(session.state["preferences"]["languages"]), 2)

    async def test_rejects_csharp_and_invalid_session_snapshots(self):
        module = load_sample("AgentWithSerialization")
        for snapshot in ({"stateBag": {"InMemoryChatHistoryProvider": {"messages": []}}}, [],
                         {"type": "session", "session_id": "", "state": {}},
                         {"type": "session", "session_id": "id", "state": []}):
            with self.subTest(snapshot=snapshot):
                path = self.temp / "invalid.json"
                path.write_text(json.dumps(snapshot), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Python AgentSession"):
                    module.load_session(path)
        path.write_text("not json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            module.load_session(path)
        with self.assertRaises(FileNotFoundError):
            module.load_session(self.temp / "missing.json")

    async def test_serialization_write_failure_is_not_silently_ignored(self):
        module = load_sample("AgentWithSerialization")
        sdk, requests = self.configured_sdk()
        with self.assertRaises(OSError):
            await module.run_example(self.chat_agent(sdk), path=self.temp / "missing" / "session.json")
        self.assertEqual(len(requests), 1)

    async def test_all_original_multimodal_assets_are_encoded_with_correct_types(self):
        module = load_sample("AgentWithChatMessagesAndMultiModal")
        for asset, mime in module.ASSET_MEDIA_TYPES.items():
            with self.subTest(asset=asset):
                sdk, requests = self.configured_sdk()
                response = await module.run_example(self.chat_agent(sdk), asset=asset)
                messages = requests[0]["body"]["messages"]
                # 1.8.1 splits the text and media parts into adjacent user messages.
                self.assertEqual([m["role"] for m in messages], ["system", "user", "user"])
                media = messages[-1]["content"][0]
                self.assertEqual(response.text, MOVES)
                original = (module.DATA_DIR / asset).read_bytes()
                if asset == "Task.mp3":
                    self.assertEqual(media["type"], "input_audio")
                    self.assertEqual(media["input_audio"]["format"], "mp3")
                    self.assertEqual(base64.b64decode(media["input_audio"]["data"]), original)
                    self.assertEqual(wire_text(messages[-2]), "Follow the spoken task safely.")
                else:
                    self.assertEqual(wire_text(messages[-2]), "Look at the image of the map and proceed safely.")
                    self.assertEqual(media["type"], "image_url")
                    url = media["image_url"]["url"]
                    prefix, encoded = url.split(",", 1)
                    self.assertEqual(prefix, f"data:{mime};base64")
                    self.assertEqual(base64.b64decode(encoded), original)
                    if asset.endswith(".png"):
                        self.assertTrue(original.startswith(b"\x89PNG\r\n\x1a\n"))
                    else:
                        self.assertTrue(original.startswith(b"\xff\xd8"))

    async def test_multimodal_default_only_needs_map_and_missing_or_invalid_asset_fails(self):
        module = load_sample("AgentWithChatMessagesAndMultiModal")
        (self.temp / "Map.png").write_bytes(b"offline-map")
        sdk, requests = self.configured_sdk()
        await module.run_example(self.chat_agent(sdk), data_dir=self.temp)
        self.assertEqual(len(requests), 1)
        self.assertEqual(module.DATA_DIR, Path(module.__file__).parent / "Data")
        with self.assertRaisesRegex(ValueError, "Unknown asset"):
            module.build_conversation(self.temp, asset="..\\secret.txt")
        with self.assertRaises(FileNotFoundError):
            await module.run_example(self.chat_agent(sdk), data_dir=self.temp, asset="path.jpg")
        self.assertEqual(len(requests), 1)

    async def test_audio_main_uses_azure_audio_credentials_and_deployment(self):
        module = load_sample("AgentWithChatMessagesAndMultiModal")
        sdk, requests = self.configured_sdk(provider="azure", audio=True)
        with patch.object(module, "create_sdk_client", return_value=sdk) as factory:
            await module.main("azure", asset="Task.mp3")
        factory.assert_called_once_with("azure", audio=True)
        self.assertTrue(sdk.is_closed())
        self.assertEqual(requests[0]["host"], "audio.invalid")
        self.assertEqual(requests[0]["body"]["model"], "offline-audio-model")
        self.assertEqual(requests[0]["body"]["messages"][-1]["content"][0]["type"], "input_audio")

    async def test_structured_response_preserves_schema_options_and_nullable_values(self):
        module = load_sample("AgentWithStructuredOutput")
        sdk, requests = self.configured_sdk(lambda record, index: completion_payload(STRUCTURED))
        response = await module.run_example(module.create_agent(chat_clients.create_completion_client(sdk)))
        self.assertIsInstance(response.value, module.StepsResponse)
        self.assertEqual(response.value.Steps[0].Move, "stop")
        self.assertIsNone(response.value.Steps[0].Value)
        self.assertEqual(response.value.Steps[1].Value, "5")
        self.assertEqual(json.loads(self.output.getvalue()), json.loads(STRUCTURED))
        body = requests[0]["body"]
        self.assertEqual(body["temperature"], 0.4)
        self.assertEqual(body["max_completion_tokens"], 300)
        schema = body["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["required"], ["Steps"])
        step = schema["schema"]["$defs"]["StepItem"]
        self.assertEqual(step["required"], ["Move", "Value"])
        self.assertFalse(step["additionalProperties"])
        self.assertIn({"type": "null"}, step["properties"]["Value"]["anyOf"])

    async def test_structured_response_rejects_invalid_json_and_schema(self):
        module = load_sample("AgentWithStructuredOutput")
        for text in ("not-json", '{"Steps":[{"Move":"stop"}]}',
                     '{"Steps":[{"Move":"stop","Value":5}]}', '{"Steps":[],"extra":true}'):
            with self.subTest(text=text):
                sdk, requests = self.configured_sdk(lambda record, index: completion_payload(text))
                agent = module.create_agent(chat_clients.create_completion_client(sdk))
                with self.assertRaises((ValidationError, ValueError, ChatClientException)):
                    await module.run_example(agent)
                self.assertEqual(len(requests), 1)

    async def test_streaming_prints_deltas_once_and_retains_usage(self):
        sdk, requests = self.configured_sdk(lambda record, index: streaming_payload())
        agent = self.chat_agent(sdk)
        response = await load_sample("AgentWithStreaming").run_example(agent)
        self.assertEqual(response.text, MOVES)
        self.assertEqual(response.usage_details, {"input_token_count": 23, "output_token_count": 7,
                                                "total_token_count": 30})
        output = self.output.getvalue()
        self.assertEqual(output.count(MOVES), 1)
        self.assertIn("Input Tokens: 23", output)
        self.assertIn("Output Tokens: 7", output)
        self.assertIn("Total Tokens: 30", output)
        self.assertTrue(requests[0]["body"]["stream"])
        self.assertTrue(requests[0]["body"]["stream_options"]["include_usage"])
        self.assertEqual(agent.context_providers, [])

    async def test_shared_session_auditor_sees_motors_moves_but_own_instructions(self):
        module = load_sample("MultipleAgentsWithSameSession")
        audit_text = "Turning and moving avoids the tree while returning to the original path."
        sdk, requests = self.configured_sdk(lambda record, index: completion_payload(MOVES if index == 0 else audit_text))
        motors, auditor = module.create_agents(chat_clients.create_completion_client(sdk))
        first, audit, session = await module.run_example(motors, auditor)
        body = requests[1]["body"]
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(wire_text(body["messages"][0]), agent_examples.AUDITOR_INSTRUCTIONS)
        self.assertEqual(wire_text(body["messages"][2]), first.text)
        self.assertEqual(wire_text(body["messages"][-1]), agent_examples.AUDIT_PROMPT)
        self.assertEqual(audit.text, audit_text)
        history = session.state["shared_history"]["messages"]
        self.assertEqual([m.role for m in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual([m.author_name for m in history if m.role == "assistant"], ["MotorsAgent", "AuditorAgent"])
        self.assertEqual(set(session.state), {"shared_history"})

    async def test_background_continuations_poll_gets_without_duplicate_inputs(self):
        module = load_sample("AgentWithBackgroundResponses")
        sdk, requests = self.configured_sdk(lambda record, index: responses_payload(("queued", "in_progress", "completed")[index]))
        with patch.object(module.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            response, session = await module.run_example(self.responses_agent(sdk), max_attempts=2, delay=0.25)
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual([r["method"] for r in requests], ["POST", "GET", "GET"])
        self.assertEqual([r["path"] for r in requests], ["/v1/responses", "/v1/responses/resp_offline", "/v1/responses/resp_offline"])
        self.assertTrue(requests[0]["body"]["background"])
        self.assertTrue(requests[0]["body"]["store"])
        self.assertEqual(wire_text(requests[0]["body"]["input"][-1]), agent_examples.TREE_PROMPT)
        self.assertTrue(all(r["body"] is None for r in requests[1:]))
        self.assertEqual(response.text, MOVES)
        self.assertIsNone(response.continuation_token)
        self.assertEqual(session.service_session_id, "resp_offline")
        self.assertEqual(session.state, {})
        output = self.output.getvalue()
        self.assertIn("Response Status: queued", output)
        self.assertIn("Response Status: completed", output)
        self.assertIn("Has Continuation Token: True", output)
        self.assertIn("Has Continuation Token: False", output)

    async def test_background_immediate_completion_does_not_sleep_or_poll(self):
        module = load_sample("AgentWithBackgroundResponses")
        sdk, requests = self.configured_sdk(lambda record, index: responses_payload())
        with patch.object(module.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            response, _ = await module.run_example(self.responses_agent(sdk), max_attempts=1, delay=0)
        sleep.assert_not_awaited()
        self.assertEqual(len(requests), 1)
        self.assertEqual(response.text, MOVES)

    async def test_background_timeout_is_bounded_and_identifies_pending_response(self):
        module = load_sample("AgentWithBackgroundResponses")
        sdk, requests = self.configured_sdk(lambda record, index: responses_payload("in_progress"))
        with patch.object(module.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaisesRegex(TimeoutError, "after 2 attempts.*resp_offline"):
                await module.run_example(self.responses_agent(sdk), max_attempts=2, delay=0)
        self.assertEqual(len(requests), 3)
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(requests[-1]["method"], "GET")

    async def test_background_invalid_polling_parameters_fail_before_requests(self):
        module = load_sample("AgentWithBackgroundResponses")
        sdk, requests = self.configured_sdk()
        for attempts, delay in ((0, 0), (-1, 1), (1, -0.1)):
            with self.subTest(attempts=attempts, delay=delay):
                with self.assertRaisesRegex(ValueError, "positive.*nonnegative"):
                    await module.run_example(self.responses_agent(sdk), max_attempts=attempts, delay=delay)
        self.assertEqual(requests, [])

    async def test_background_terminal_failures_are_not_reported_as_success(self):
        module = load_sample("AgentWithBackgroundResponses")
        for status in ("failed", "cancelled", "incomplete"):
            for initially_pending in (False, True):
                with self.subTest(status=status, pending=initially_pending):
                    sdk, requests = self.configured_sdk(lambda record, index: responses_payload(
                        "queued" if initially_pending and index == 0 else status,
                    ))
                    with self.assertRaisesRegex(RuntimeError, f"status {status}"):
                        await module.run_example(self.responses_agent(sdk), max_attempts=1, delay=0)
                    self.assertEqual(len(requests), 2 if initially_pending else 1)

    def conversation_responder(self, *, fail=None):
        response_count = 0

        def responder(record, index):
            nonlocal response_count
            path = record["path"]
            if path.endswith("/conversations"):
                return {"id": "conv_offline", "object": "conversation", "created_at": 1760000000, "metadata": {}}
            if record["method"] == "DELETE":
                return {"id": "conv_offline", "object": "conversation.deleted", "deleted": True}
            if path.endswith("/items"):
                if fail == "history":
                    return httpx.Response(500, json={"error": {"message": "History failed", "type": "server_error"}})
                if record["query"].get("after"):
                    items = [service_item("assistant", "forward, distance 5", "msg_second")]
                    return {"object": "list", "data": items, "first_id": "msg_second", "last_id": "msg_second", "has_more": False}
                items = [service_item("user", agent_examples.TREE_PROMPT, "msg_first"),
                         {"type": "reasoning", "id": "rs_offline", "summary": []}]
                return {"object": "list", "data": items, "first_id": "msg_first", "last_id": "rs_offline", "has_more": True}
            if path.endswith("/responses"):
                response_count += 1
                if fail == f"turn{response_count}":
                    return httpx.Response(500, json={"error": {"message": "Turn failed", "type": "server_error"}})
                return responses_payload(text=MOVES if response_count == 1 else "forward, distance 5", conversation_id="conv_offline")
            raise AssertionError(f"Unexpected request: {record}")
        return responder

    async def test_service_conversation_uses_real_api_paginated_history_and_cleanup(self):
        module = load_sample("AgentWithInServiceConversation")
        sdk, requests = self.configured_sdk(self.conversation_responder())
        first, second, session = await module.run_example(sdk, self.responses_agent(sdk))
        self.assertEqual(first.text, MOVES)
        self.assertEqual(second.text, "forward, distance 5")
        self.assertEqual(session.service_session_id, "conv_offline")
        self.assertEqual(session.state, {})
        self.assertEqual(requests[0]["body"]["items"], [{
            "type": "message", "role": "system", "content": agent_examples.ROBOT_INSTRUCTIONS,
        }])
        turns = [r for r in requests if r["path"].endswith("/responses")]
        for index, turn in enumerate(turns):
            body = turn["body"]
            self.assertEqual(body["conversation"], "conv_offline")
            self.assertNotIn("previous_response_id", body)
            self.assertEqual(len(body["input"]), 1)
            self.assertEqual(wire_text(body["input"][0]), (agent_examples.TREE_PROMPT, agent_examples.FOLLOWUP_PROMPT)[index])
        pages = [r for r in requests if r["path"].endswith("/items")]
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["query"]["order"], "asc")
        self.assertEqual(pages[1]["query"]["after"], "rs_offline")
        self.assertEqual((requests[-1]["method"], requests[-1]["path"]), ("DELETE", "/v1/conversations/conv_offline"))
        output = self.output.getvalue()
        self.assertIn("user [msg_first]", output)
        self.assertIn("assistant [msg_second]", output)
        self.assertNotIn("rs_offline", output)
        self.assertIn("Deleted: True", output)

    async def test_service_conversation_cleanup_runs_after_turn_and_history_failures(self):
        module = load_sample("AgentWithInServiceConversation")
        for failure in ("turn1", "turn2", "history"):
            with self.subTest(failure=failure):
                sdk, requests = self.configured_sdk(self.conversation_responder(fail=failure))
                error_type = InternalServerError if failure == "history" else ChatClientException
                with self.assertRaises(error_type):
                    await module.run_example(sdk, self.responses_agent(sdk))
                self.assertEqual((requests[-1]["method"], requests[-1]["path"]), ("DELETE", "/v1/conversations/conv_offline"))
                self.assertEqual(sum(r["method"] == "DELETE" for r in requests), 1)

    async def test_service_conversation_creation_failure_does_not_delete_unknown_id(self):
        module = load_sample("AgentWithInServiceConversation")
        sdk, requests = self.configured_sdk(lambda record, index: httpx.Response(
            500, json={"error": {"message": "Creation failed", "type": "server_error"}},
        ))
        with self.assertRaises(InternalServerError):
            await module.run_example(sdk, self.responses_agent(sdk))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["method"], "POST")

    async def test_all_ten_main_functions_use_factories_and_close_sdk(self):
        for provider in ("openai", "azure"):
            for name in PROJECTS:
                with self.subTest(provider=provider, project=name):
                    module = load_sample(name)
                    if name == "AgentWithInServiceConversation":
                        responder = self.conversation_responder()
                    elif name == "AgentWithBackgroundResponses":
                        responder = lambda record, index: responses_payload()
                    elif name == "AgentWithStreaming":
                        responder = lambda record, index: streaming_payload()
                    elif name == "AgentWithStructuredOutput":
                        responder = lambda record, index: completion_payload(STRUCTURED)
                    else:
                        responder = lambda record, index: completion_payload()
                    sdk, requests = self.configured_sdk(responder, provider=provider)
                    session_path = self.temp / f"{provider}_session.json"
                    with patch.object(module, "create_sdk_client", return_value=sdk) as factory:
                        if name == "AgentWithSerialization":
                            with patch.object(module, "SESSION_PATH", session_path):
                                await module.main(provider)
                        else:
                            await module.main(provider)
                    factory.assert_called_once_with(provider, **({"audio": False} if name == "AgentWithChatMessagesAndMultiModal" else {}))
                    self.assertTrue(sdk.is_closed())
                    self.assertGreater(len(requests), 0)
                    expected_model = ENV[("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME" if provider == "azure" else "OPENAI_RESPONSES_MODEL_ID")
                                         if name in {"AgentWithBackgroundResponses", "AgentWithInServiceConversation"}
                                         else ("AZURE_OPENAI_DEPLOYMENT_NAME" if provider == "azure" else "OPENAI_CHAT_MODEL_ID")]
                    for request in requests:
                        if request["body"] and "model" in request["body"]:
                            self.assertEqual(request["body"]["model"], expected_model)
                    if name == "AgentWithSerialization":
                        self.assertTrue(session_path.is_file())
                        self.assertEqual(json.loads(session_path.read_text(encoding="utf-8"))["type"], "session")

    async def test_service_conversation_cleanup_runs_when_agent_is_cancelled(self):
        import asyncio

        module = load_sample("AgentWithInServiceConversation")
        sdk, requests = self.configured_sdk(self.conversation_responder())
        agent = self.responses_agent(sdk)
        with patch.object(agent, "run", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await module.run_example(sdk, agent)
        self.assertEqual([r["method"] for r in requests], ["POST", "DELETE"])
        self.assertEqual(requests[-1]["path"], "/v1/conversations/conv_offline")

    async def test_service_conversation_deletion_failure_is_visible_and_sdk_closes(self):
        module = load_sample("AgentWithInServiceConversation")
        normal = self.conversation_responder()

        def responder(record, index):
            if record["method"] == "DELETE":
                return httpx.Response(500, json={"error": {"message": "Deletion failed", "type": "server_error"}})
            return normal(record, index)

        sdk, requests = self.configured_sdk(responder)
        with patch.object(module, "create_sdk_client", return_value=sdk):
            with self.assertRaisesRegex(InternalServerError, "Deletion failed"):
                await module.main()
        self.assertTrue(sdk.is_closed())
        self.assertEqual(requests[-1]["method"], "DELETE")
        self.assertNotIn("Deleted: True", self.output.getvalue())

    async def test_background_retrieval_failure_propagates_instead_of_polling_forever(self):
        module = load_sample("AgentWithBackgroundResponses")

        def responder(record, index):
            if index == 0:
                return responses_payload("queued")
            return httpx.Response(500, json={"error": {"message": "Retrieval failed", "type": "server_error"}})

        sdk, requests = self.configured_sdk(responder)
        with self.assertRaises(ChatClientException):
            await module.run_example(self.responses_agent(sdk), max_attempts=5, delay=0)
        self.assertEqual([r["method"] for r in requests], ["POST", "GET"])
        self.assertNotIn("[FINAL]", self.output.getvalue())

    async def test_streaming_empty_text_keeps_usage_and_finishes(self):
        sdk, requests = self.configured_sdk(lambda record, index: streaming_payload(""))
        response = await load_sample("AgentWithStreaming").run_example(self.chat_agent(sdk))
        self.assertEqual(response.text, "")
        self.assertEqual(response.usage_details["total_token_count"], 30)
        self.assertIn("Total Tokens: 30", self.output.getvalue())
        self.assertEqual(len(requests), 1)

    async def test_main_closes_sdk_after_provider_failure(self):
        for name in ("Agent", "AgentWithStreaming", "AgentWithBackgroundResponses", "AgentWithInServiceConversation"):
            with self.subTest(project=name):
                module = load_sample(name)
                sdk, _ = self.configured_sdk(lambda record, index: httpx.Response(
                    500, json={"error": {"message": "Provider failed", "type": "server_error"}},
                ))
                with patch.object(module, "create_sdk_client", return_value=sdk):
                    error_type = InternalServerError if name == "AgentWithInServiceConversation" else ChatClientException
                    with self.assertRaises(error_type):
                        await module.main()
                self.assertTrue(sdk.is_closed())


if __name__ == "__main__":
    unittest.main()
