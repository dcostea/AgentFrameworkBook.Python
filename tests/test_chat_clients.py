"""Offline regression tests for the chapter 03/04 non-ONNX chat migrations.

Run with: python -m unittest discover -s tests -p test_chat_clients.py -v
"""

import base64
import copy
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI


ROOT = Path(__file__).resolve().parents[1]
ENV = {
    "OPENAI_API_KEY": "offline-openai-key",
    "OPENAI_CHAT_MODEL_ID": "openai-chat-model",
    "OPENAI_RESPONSES_MODEL_ID": "openai-responses-model",
    "AZURE_OPENAI_ENDPOINT": "https://azure.invalid",
    "AZURE_OPENAI_API_KEY": "offline-azure-key",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "azure-chat-deployment",
    "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME": "azure-responses-deployment",
    "AZURE_OPENAI_AUDIO_ENDPOINT": "https://audio.invalid",
    "AZURE_OPENAI_AUDIO_API_KEY": "offline-audio-key",
    "AZURE_OPENAI_AUDIO_DEPLOYMENT_NAME": "azure-audio-deployment",
    "ANTHROPIC_API_KEY": "offline-anthropic-key",
    "ANTHROPIC_MODEL_ID": "claude-offline-model",
}
SAMPLES = (
    ("03.ChatClients", "AnthropicChatClient"),
    ("03.ChatClients", "AzureOpenAIChatClient"),
    ("03.ChatClients", "AzureOpenAIChatClientWithImages"),
    ("03.ChatClients", "Chatbot"),
    ("03.ChatClients", "ChatbotWithChatOptions"),
    ("03.ChatClients", "ChatbotWithStreaming"),
    ("03.ChatClients", "ChatbotWithTools"),
    ("03.ChatClients", "ChatClientWithMiddleware"),
    ("03.ChatClients", "OpenAIChatClient"),
    ("04.ChatClients", "AzureOpenAIResponses"),
    ("04.ChatClients", "AzureOpenAIResponsesWithBackgroundResponses"),
    ("04.ChatClients", "OllamaChatClient"),
    ("04.ChatClients", "OpenAIResponses"),
    ("04.ChatClients", "OpenAIResponsesWithBackgroundResponses"),
)

# Importing the production helper must not load a developer's root .env file.
with patch.dict(os.environ, ENV, clear=True), patch("dotenv.load_dotenv", return_value=False):
    sys.path.insert(0, str(ROOT))
    from agent_framework import ChatContext, ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream
    from agent_framework.exceptions import ChatClientException
    from shared import background_responses, chat_clients, chat_middleware, chatbot


def load_sample(chapter, name):
    path = ROOT / chapter / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"offline_{chapter[:2]}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def completion_payload(text='["turn right", "forward"]', *, tool_calls=None, response_id="chatcmpl-offline"):
    message = {"role": "assistant", "content": None if tool_calls else text, "refusal": None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1760000000,
        "model": "offline-served-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 23, "completion_tokens": 7, "total_tokens": 30},
    }


def responses_payload(status="completed", *, response_id="resp-offline-job", text="Avoided the tree."):
    completed = status == "completed"
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1760000000,
        "model": "offline-served-model",
        "status": status,
        "background": True,
        "store": True,
        "error": {"code": "server_error", "message": "Offline provider failure"} if status == "failed" else None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": None,
        "max_output_tokens": 512,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 1.0,
        "top_p": 1.0,
        "output": [{
            "id": "msg-offline",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }] if completed else [],
        "usage": {
            "input_tokens": 23,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 30,
        } if completed else None,
    }


def streaming_response(text, response_id="chatcmpl-offline-stream"):
    chunks = []
    for index, part in enumerate((text[:5], "", text[5:])):
        chunks.append({
            "id": response_id, "object": "chat.completion.chunk", "created": 1760000000,
            "model": "offline-served-model",
            "choices": [{"index": 0, "delta": {"content": part, **({"role": "assistant"} if index == 0 else {})},
                         "finish_reason": None}],
        })
    chunks.append({
        "id": response_id, "object": "chat.completion.chunk", "created": 1760000000,
        "model": "offline-served-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    chunks.append({
        "id": response_id, "object": "chat.completion.chunk", "created": 1760000000,
        "model": "offline-served-model", "choices": [],
        "usage": {"prompt_tokens": 23, "completion_tokens": 7, "total_tokens": 30},
    })
    data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=data.encode(), headers={"Content-Type": "text/event-stream"})


def wire_text(message):
    contents = message.get("content", "")
    if isinstance(contents, str):
        return contents
    return "".join(item.get("text", "") for item in contents or [])


def async_sdk_context():
    sdk = Mock()
    sdk.__aenter__ = AsyncMock(return_value=sdk)
    sdk.__aexit__ = AsyncMock(return_value=False)
    return sdk


class OfflineChatClientTests(unittest.IsolatedAsyncioTestCase):
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

    def configured_sdk(self, responder, *, provider="openai", audio=False):
        requests = []

        def handle(request):
            record = {
                "method": request.method, "host": request.url.host, "path": request.url.path,
                "query": dict(request.url.params),
                "body": json.loads(request.content) if request.content else None,
                "authorization": request.headers.get("authorization"),
                "api-key": request.headers.get("api-key"),
            }
            requests.append(record)
            payload = responder(record, len(requests) - 1)
            return payload if isinstance(payload, httpx.Response) else httpx.Response(200, json=payload)

        def construct(cls, **kwargs):
            sdk = cls(
                **kwargs, max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
            )
            self.addAsyncCleanup(sdk.close)
            return sdk

        with patch.object(chat_clients, "AsyncOpenAI", side_effect=lambda **kw: construct(AsyncOpenAI, **kw)), \
             patch.object(chat_clients, "AsyncAzureOpenAI", side_effect=lambda **kw: construct(AsyncAzureOpenAI, **kw)):
            sdk = chat_clients.create_sdk_client(provider, audio=audio)
        return sdk, requests

    def record_framework_calls(self, client):
        calls = []
        original = client.get_response

        def record(messages, **kwargs):
            # The framework can mutate continuation options after completion.
            calls.append((list(messages), copy.deepcopy(kwargs)))
            return original(messages, **kwargs)

        self.enterContext(patch.object(client, "get_response", side_effect=record))
        return calls

    def test_all_fourteen_samples_import_without_configuration_or_requests(self):
        self.assertEqual(len(SAMPLES), 14)
        with patch.dict(os.environ, {}, clear=True):
            for chapter, name in SAMPLES:
                with self.subTest(chapter=chapter, sample=name):
                    module = load_sample(chapter, name)
                    self.assertTrue(inspect.iscoroutinefunction(module.main))
                    self.assertEqual(module.__file__, str(ROOT / chapter / name / "main.py"))
        self.assertEqual(self.output.getvalue(), "")

    async def test_azure_v1_normalization_routes_each_suffix_once(self):
        for suffix in ("", "/", "/openai", "/openai/", "/openai/v1", "/openai/v1/"):
            with self.subTest(suffix=suffix), patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://azure.invalid" + suffix}):
                self.assertEqual(chat_clients.azure_base_url("https://azure.invalid" + suffix), "https://azure.invalid/openai/v1/")
                sdk, requests = self.configured_sdk(lambda *_: completion_payload(), provider="azure")
                self.assertIs(type(sdk), AsyncOpenAI)
                self.assertEqual(str(sdk.base_url), "https://azure.invalid/openai/v1/")
                response = await chat_clients.create_completion_client(sdk, "azure").get_response([Message("user", ["Drive."])])
                self.assertEqual(response.text, '["turn right", "forward"]')
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["path"], "/openai/v1/chat/completions")
                self.assertEqual(requests[0]["query"], {})
                self.assertEqual(requests[0]["body"]["model"], "azure-chat-deployment")
                self.assertEqual(requests[0]["authorization"], "Bearer offline-azure-key")

    async def test_legacy_azure_api_version_and_audio_precedence(self):
        cases = (
            (False, "", {"AZURE_OPENAI_API_VERSION": "2025-01-01-preview"}, "2025-01-01-preview"),
            (False, "/openai/v1/", {"AZURE_OPENAI_API_VERSION": "2025-01-01-preview"}, "2025-01-01-preview"),
            (True, "/openai/", {"AZURE_OPENAI_API_VERSION": "2025-01-01-preview"}, "2025-01-01-preview"),
            (True, "/openai/v1", {"AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
                                   "AZURE_OPENAI_AUDIO_API_VERSION": "2025-02-01-preview"}, "2025-02-01-preview"),
        )
        for audio, suffix, versions, expected_version in cases:
            with self.subTest(audio=audio, suffix=suffix, version=expected_version):
                prefix = "AZURE_OPENAI_AUDIO" if audio else "AZURE_OPENAI"
                host = "audio.invalid" if audio else "azure.invalid"
                with patch.dict(os.environ, {**versions, f"{prefix}_ENDPOINT": f"https://{host}{suffix}"}):
                    sdk, requests = self.configured_sdk(lambda *_: completion_payload(), provider="azure", audio=audio)
                    self.assertIs(type(sdk), AsyncAzureOpenAI)
                    await chat_clients.create_completion_client(sdk, "azure", audio=audio).get_response([Message("user", ["Drive."])])
                self.assertEqual(len(requests), 1)
                request = requests[0]
                model = "azure-audio-deployment" if audio else "azure-chat-deployment"
                self.assertEqual(request["path"], f"/openai/deployments/{model}/chat/completions")
                self.assertEqual(request["query"], {"api-version": expected_version})
                self.assertEqual(request["host"], host)
                self.assertEqual(request["api-key"], "offline-audio-key" if audio else "offline-azure-key")

    async def test_provider_selection_base_url_override_and_request_model_override(self):
        for provider, model, host, path, key in (
            ("openai", "openai-chat-model", "proxy.invalid", "/custom/v1/chat/completions", "offline-openai-key"),
            ("azure", "azure-chat-deployment", "azure.invalid", "/openai/v1/chat/completions", "offline-azure-key"),
        ):
            with self.subTest(provider=provider), patch.dict(os.environ, {"OPENAI_BASE_URL": "https://proxy.invalid/custom/v1/"}):
                sdk, requests = self.configured_sdk(lambda *_: completion_payload(), provider=provider)
                client = chat_clients.create_completion_client(sdk, provider)
                self.assertIs(client.client, sdk)
                self.assertEqual(client.model, model)
                await client.get_response([Message("user", ["Drive."])])
                result = await client.get_response([Message("user", ["Park."])], options={"model": "per-request-model"})
                self.assertEqual(result.model, "offline-served-model")
                self.assertEqual([r["body"]["model"] for r in requests], [model, "per-request-model"])
                for request in requests:
                    self.assertEqual((request["host"], request["path"]), (host, path))
                    self.assertEqual(request["authorization"], f"Bearer {key}")

    def test_model_configuration_fallbacks_and_required_environment_messages(self):
        for provider, flags, expected in (
            ("openai", {}, "openai-chat-model"),
            ("openai", {"responses": True}, "openai-responses-model"),
            ("azure", {}, "azure-chat-deployment"),
            ("azure", {"responses": True}, "azure-responses-deployment"),
            ("azure", {"audio": True, "responses": True}, "azure-audio-deployment"),
        ):
            with self.subTest(provider=provider, flags=flags):
                self.assertEqual(chat_clients.model_for(provider, **flags), expected)
        fallback_env = {key: value for key, value in ENV.items() if key not in {
            "OPENAI_CHAT_MODEL_ID", "OPENAI_RESPONSES_MODEL_ID", "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME",
        }}
        with patch.dict(os.environ, fallback_env, clear=True):
            self.assertEqual(chat_clients.model_for(), "gpt-5.4-mini")
            self.assertEqual(chat_clients.model_for(responses=True), "gpt-5.4-mini")
            self.assertEqual(chat_clients.model_for("azure", responses=True), "azure-chat-deployment")
        for chat_model, expected in (("", "gpt-5.4-mini"), ("openai-chat-model", "openai-chat-model")):
            with self.subTest(blank_responses_model=True, chat_model=chat_model), patch.dict(os.environ, {
                "OPENAI_RESPONSES_MODEL_ID": "", "OPENAI_CHAT_MODEL_ID": chat_model,
            }):
                self.assertEqual(chat_clients.model_for(responses=True), expected)
        for name, operation in (
            ("OPENAI_API_KEY", lambda: chat_clients.create_sdk_client()),
            ("AZURE_OPENAI_ENDPOINT", lambda: chat_clients.create_sdk_client("azure")),
            ("AZURE_OPENAI_API_KEY", lambda: chat_clients.create_sdk_client("azure")),
            ("AZURE_OPENAI_AUDIO_API_KEY", lambda: chat_clients.create_sdk_client("azure", audio=True)),
            ("AZURE_OPENAI_AUDIO_DEPLOYMENT_NAME", lambda: chat_clients.model_for("azure", audio=True)),
            ("AZURE_OPENAI_DEPLOYMENT_NAME", lambda: chat_clients.model_for("azure")),
        ):
            for value in (None, " \t "):
                with self.subTest(missing=name, value=value), patch.dict(os.environ, ENV, clear=True):
                    if value is None:
                        os.environ.pop(name)
                    else:
                        os.environ[name] = value
                    with self.assertRaises(RuntimeError) as error:
                        operation()
                    self.assertEqual(str(error.exception), f"{name} is not set. Configure it in the root .env file or environment.")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "  trimmed-key \t"}):
            self.assertEqual(chat_clients.require_env("OPENAI_API_KEY"), "trimmed-key")
        for operation in (chat_clients.create_sdk_client, chat_clients.model_for):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "Unknown provider: unknown"):
                operation("unknown")

    async def test_sdk_and_framework_comparison_samples_use_the_intended_api(self):
        for chapter, name, provider, responses in (
            ("03.ChatClients", "OpenAIChatClient", "openai", False),
            ("03.ChatClients", "AzureOpenAIChatClient", "azure", False),
            ("04.ChatClients", "OpenAIResponses", "openai", True),
            ("04.ChatClients", "AzureOpenAIResponses", "azure", True),
        ):
            with self.subTest(sample=name):
                module = load_sample(chapter, name)
                sdk, requests = self.configured_sdk(
                    lambda *_: responses_payload(text="Comparison result.") if responses else completion_payload("Comparison result."),
                    provider=provider,
                )
                start = self.output.tell()
                with patch.object(module, "create_sdk_client", return_value=sdk) as factory:
                    await module.main()
                factory.assert_called_once_with(*(("azure",) if provider == "azure" else ()))
                self.assertTrue(sdk.is_closed())
                self.assertEqual(len(requests), 2)
                prefix = "/openai/v1/" if provider == "azure" else "/v1/"
                expected_model = chat_clients.model_for(provider, responses=responses)
                for request in requests:
                    self.assertEqual((request["method"], request["path"]), ("POST", prefix + ("responses" if responses else "chat/completions")))
                    self.assertEqual(request["body"]["model"], expected_model)
                if responses:
                    self.assertEqual(requests[0]["body"]["input"], chat_clients.ROBOT_PROMPT)
                    self.assertEqual(requests[1]["body"]["input"][0]["role"], "user")
                    self.assertEqual(requests[1]["body"]["input"][0]["content"][0]["text"], chat_clients.ROBOT_PROMPT)
                else:
                    for request in requests:
                        self.assertEqual(len(request["body"]["messages"]), 1)
                        self.assertEqual(wire_text(request["body"]["messages"][0]), chat_clients.ROBOT_PROMPT)
                self.assertEqual(self.output.getvalue()[start:].count("Comparison result."), 2)

    async def test_multimodal_sample_defaults_to_mp3_and_formats_images(self):
        module = load_sample("03.ChatClients", "AzureOpenAIChatClientWithImages")
        data = b"offline-media\x00\xff"
        encoded = base64.b64encode(data).decode()
        for mode in ("audio", "image"):
            with self.subTest(mode=mode):
                sdk, requests = self.configured_sdk(lambda *_: completion_payload(), provider="azure", audio=mode == "audio")
                with patch.object(module, "create_sdk_client", return_value=sdk) as factory, \
                     patch.object(module, "create_completion_client", wraps=chat_clients.create_completion_client) as connector, \
                     patch.object(Path, "read_bytes", autospec=True, return_value=data) as read_bytes:
                    if mode == "audio":
                        await module.main()
                    else:
                        await module.main("image")
                factory.assert_called_once_with("azure", audio=mode == "audio")
                connector.assert_called_once_with(sdk, "azure", audio=mode == "audio")
                self.assertEqual(read_bytes.call_args.args[0], module.DATA_DIR / ("Task.mp3" if mode == "audio" else "path.jpg"))
                self.assertEqual(read_bytes.call_count, 1)
                self.assertTrue(sdk.is_closed())
                self.assertEqual(len(requests), 1)
                body = requests[0]["body"]
                self.assertEqual(wire_text(body["messages"][0]), chat_clients.ROBOT_SYSTEM)
                self.assertEqual(body["messages"][0]["role"], "system")
                parts = body["messages"][1]["content"]
                if mode == "audio":
                    self.assertEqual(body["model"], "azure-audio-deployment")
                    self.assertEqual(requests[0]["host"], "audio.invalid")
                    self.assertEqual(parts, [{"type": "input_audio", "input_audio": {"data": encoded, "format": "mp3"}}])
                else:
                    self.assertEqual(body["model"], "azure-chat-deployment")
                    # This adapter emits text and image as separate user messages.
                    self.assertEqual([m["role"] for m in body["messages"]], ["system", "user", "user"])
                    self.assertEqual(parts, "Look at the image and proceed safely.")
                    image_parts = body["messages"][2]["content"]
                    self.assertEqual(len(image_parts), 1)
                    self.assertEqual(image_parts[0]["type"], "image_url")
                    self.assertEqual(image_parts[0]["image_url"]["url"], f"data:image/jpeg;base64,{encoded}")
        with patch.object(Path, "read_bytes") as read_bytes, self.assertRaisesRegex(ValueError, "mode must be audio or image"):
            module.build_messages("video")
        read_bytes.assert_not_called()

    async def test_chat_options_translate_limits_tools_and_steps_schema(self):
        module = load_sample("03.ChatClients", "ChatbotWithChatOptions")
        self.assertEqual(module.chat_options()["max_tokens"], 512)
        for parallel, include_tools, overrides, limit in (
            (True, True, {}, 512),
            (False, True, {"CHATBOT_MAX_OUTPUT_TOKENS": "5"}, 5),
            (True, False, {"CHATBOT_MAX_OUTPUT_TOKENS": "1024"}, 1024),
        ):
            with self.subTest(parallel=parallel, tools=include_tools, limit=limit), patch.dict(os.environ, overrides):
                options = module.chat_options()
                options.update(tool_choice="auto", allow_multiple_tool_calls=parallel)
                if include_tools:
                    options["tools"] = chatbot.motor_functions()
                sdk, requests = self.configured_sdk(lambda *_: completion_payload('{"Steps":["turn right","forward"]}'), provider="azure")
                response = await chat_clients.create_completion_client(sdk, "azure").get_response([Message("user", ["Drive."])], options=options)
                self.assertEqual(response.value.steps, ["turn right", "forward"])
                self.assertEqual(response.finish_reason, "stop")
                self.assertEqual(response.response_id, "chatcmpl-offline")
                self.assertEqual(len(requests), 1)
                body = requests[0]["body"]
                self.assertEqual(body["max_completion_tokens"], limit)
                self.assertNotIn("max_tokens", body)
                self.assertNotIn("allow_multiple_tool_calls", body)
                for key, expected in {"temperature": 0.4, "top_p": 0.9, "frequency_penalty": 0.5,
                                      "presence_penalty": 0.3, "stop": ["END"]}.items():
                    self.assertEqual(body[key], expected)
                schema = body["response_format"]
                self.assertEqual(schema["type"], "json_schema")
                self.assertTrue(schema["json_schema"]["strict"])
                shape = schema["json_schema"]["schema"]
                self.assertEqual(shape["required"], ["Steps"])
                self.assertEqual(shape["properties"]["Steps"]["items"], {"type": "string"})
                self.assertNotIn("steps", shape["properties"])
                if include_tools:
                    self.assertEqual(body["parallel_tool_calls"], parallel)
                    self.assertEqual(body["tool_choice"], "auto")
                    self.assertEqual({tool["function"]["name"] for tool in body["tools"]}, {"forward", "backward", "stop", "turn_left", "turn_right"})
                else:
                    self.assertNotIn("parallel_tool_calls", body)
                    self.assertNotIn("tool_choice", body)
                self.assertEqual(options["max_tokens"], limit)
                self.assertIs(options["response_format"], chatbot.StepsResponse)

    async def test_console_two_turn_history_and_stream_output_do_not_duplicate(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                snapshots = []
                client = Mock()
                answers = ("First move.", "Second move.")

                def respond(messages, **kwargs):
                    snapshots.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
                    answer = answers[len(snapshots) - 1]
                    if not stream:
                        async def completed():
                            return ChatResponse(messages=[Message("assistant", [answer])])
                        return completed()

                    async def updates():
                        yield ChatResponseUpdate(role="assistant", message_id=f"turn-{len(snapshots)}")
                        for part in (answer[:6], answer[6:]):
                            yield ChatResponseUpdate(role="assistant", message_id=f"turn-{len(snapshots)}", contents=[Content.from_text(part)])
                    return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

                client.get_response.side_effect = respond
                read_input = Mock(side_effect=["Avoid the tree.", "Park now.", ""])
                options = {"max_tokens": 64}
                start = self.output.tell()
                history = await chatbot.run_chatbot(client, system="Robot system.", options=options, stream=stream, read_input=read_input)
                self.assertEqual([message.role for message in history], ["system", "user", "assistant", "user", "assistant"])
                self.assertEqual([message.text for message in history], [
                    "Robot system.", '## Context\nComplex command:\n"Avoid the tree."', answers[0],
                    '## Context\nComplex command:\n"Park now."', answers[1],
                ])
                self.assertEqual([len(messages) for messages, _ in snapshots], [2, 4])
                self.assertEqual([m.text for m in snapshots[1][0]], [m.text for m in history[:4]])
                for _, kwargs in snapshots:
                    self.assertEqual(kwargs["options"], {"max_tokens": 64})
                    self.assertNotIn("compaction_strategy", kwargs)
                    self.assertEqual(kwargs.get("stream", False), stream)
                self.assertEqual(options, {"max_tokens": 64})
                self.assertEqual(read_input.call_args_list, [call("User: ")] * 3)
                rendered = self.output.getvalue()[start:]
                self.assertEqual(rendered.count("Assistant: First move."), 1)
                self.assertEqual(rendered.count("Assistant: Second move."), 1)
                self.assertNotIn("First move.Second move.", rendered)
                self.assertEqual(client.get_response.call_count, 2)

    async def test_console_empty_or_eof_input_never_sends_request(self):
        for stream in (False, True):
            for ending in ("", EOFError()):
                with self.subTest(stream=stream, ending=type(ending).__name__):
                    client = Mock()
                    read_input = Mock(side_effect=[ending])
                    history = await chatbot.run_chatbot(client, stream=stream, read_input=read_input)
                    self.assertEqual([(m.role, m.text) for m in history], [("system", chat_clients.ROBOT_SYSTEM)])
                    client.get_response.assert_not_called()
                    read_input.assert_called_once_with("User: ")

    async def test_real_framework_streaming_parses_and_preserves_two_turns(self):
        sdk, requests = self.configured_sdk(lambda _, i: streaming_response(f"Move {i + 1}.", response_id=f"chatcmpl-turn-{i}"))
        client = chat_clients.create_completion_client(sdk)
        start = self.output.tell()
        history = await chatbot.run_chatbot(client, stream=True, read_input=Mock(side_effect=["Turn.", "Stop.", ""]))
        self.assertEqual([m.role for m in history], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual([m.text for m in history if m.role == "assistant"], ["Move 1.", "Move 2."])
        self.assertEqual(len(requests), 2)
        self.assertEqual([m["role"] for m in requests[1]["body"]["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual([wire_text(m) for m in requests[1]["body"]["messages"]], [m.text for m in history[:4]])
        for request in requests:
            self.assertTrue(request["body"]["stream"])
            self.assertEqual(request["body"]["stream_options"], {"include_usage": True})
        self.assertEqual(self.output.getvalue()[start:].count("Assistant: Move 1."), 1)
        self.assertEqual(self.output.getvalue()[start:].count("Assistant: Move 2."), 1)

    async def test_automatic_motor_tools_execute_and_send_matching_results(self):
        movements = (("turn_right", {"angle": 45}, "turned clockwise 45 degrees."),
                     ("turn_left", {"angle": 30}, "turned anticlockwise 30 degrees."),
                     ("stop", {}, "stopped."),
                     ("forward", {"distance": 7}, "moved forward for 7 meters."),
                     ("backward", {"distance": 2}, "moved backward for 2 meters."))
        tool_calls = [{"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
                      for name, arguments, _ in movements]
        sdk, requests = self.configured_sdk(lambda _, i: completion_payload(tool_calls=tool_calls) if i == 0 else completion_payload("Finished moving."))
        client = chat_clients.create_completion_client(sdk)
        with patch.object(chatbot.motor_tools.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            response = await client.get_response([Message("system", [chat_clients.TOOLS_SYSTEM]), Message("user", ["Avoid it."])],
                                                 options={"tools": chatbot.motor_functions(), "tool_choice": "auto", "allow_multiple_tool_calls": True})
        self.assertEqual(response.text, "Finished moving.")
        self.assertEqual(sleep.await_args_list, [call(1.0)] * 5)
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[0]["body"]["parallel_tool_calls"])
        second = requests[1]["body"]["messages"]
        assistant_calls = [message for message in second if message.get("tool_calls")]
        self.assertEqual(len(assistant_calls), 1)
        self.assertEqual(assistant_calls[0]["tool_calls"], tool_calls)
        results = {message["tool_call_id"]: message["content"] for message in second if message["role"] == "tool"}
        self.assertEqual(results, {f"call-{name}": result for name, _, result in movements})
        returned_calls = [content for m in response.messages for content in m.contents if content.type == "function_call"]
        returned_results = [content for m in response.messages for content in m.contents if content.type == "function_result"]
        self.assertEqual({c.call_id for c in returned_calls}, set(results))
        self.assertEqual({c.call_id: c.result for c in returned_results}, results)
        self.assertIn("MOTORS: Forward: 7m", self.output.getvalue())
        self.assertIn("MOTORS: TurnLeft: 30deg", self.output.getvalue())

    async def test_reduced_history_keeps_system_and_complete_tool_pairs(self):
        turns = 13

        def respond(_, index):
            if index % 2 == 0:
                return completion_payload(tool_calls=[{"id": f"call-stop-{index // 2}", "type": "function",
                                                        "function": {"name": "stop", "arguments": "{}"}}])
            return completion_payload(f"Stopped {index // 2}.")

        sdk, requests = self.configured_sdk(respond, provider="azure")
        client = chat_clients.create_completion_client(sdk, "azure")
        framework_calls = self.record_framework_calls(client)
        with patch.object(chatbot.motor_tools.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            history = await chatbot.run_chatbot(
                client, system="Keep the robot safe.", reduce_history=True,
                options={"tools": chatbot.motor_functions(), "tool_choice": "auto", "allow_multiple_tool_calls": True},
                read_input=Mock(side_effect=[f"Stop round {i}." for i in range(turns)] + [""]),
            )
        self.assertEqual(len(requests), turns * 2)
        self.assertEqual(sleep.await_count, turns)
        self.assertEqual(len(framework_calls), turns)
        for _, kwargs in framework_calls:
            strategy = kwargs["compaction_strategy"]
            self.assertEqual(strategy.keep_last_groups, 10)
            self.assertTrue(strategy.preserve_system)
        for request in requests:
            self.assertEqual(request["path"], "/openai/v1/chat/completions")
            self.assertEqual(request["body"]["model"], "azure-chat-deployment")
            self.assertTrue(request["body"]["parallel_tool_calls"])
            self.assertEqual(request["body"]["tool_choice"], "auto")
        for request in requests:
            messages = request["body"]["messages"]
            self.assertEqual((messages[0]["role"], wire_text(messages[0])), ("system", "Keep the robot safe."))
            calls = [tc["id"] for m in messages for tc in m.get("tool_calls", [])]
            results = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
            self.assertEqual(sorted(calls), sorted(results))
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(len(results), len(set(results)))
            for index, message in enumerate(messages):
                for tool_call in message.get("tool_calls", []):
                    result_index = next(i for i, m in enumerate(messages) if m.get("tool_call_id") == tool_call["id"])
                    self.assertLess(index, result_index)
        self.assertEqual(sum(m.role == "user" for m in history), turns)
        self.assertEqual(history[-1].text, "Stopped 12.")
        history_results = [content for m in history for content in m.contents if content.type == "function_result"]
        self.assertEqual(len(history_results), turns)
        self.assertEqual({content.call_id for content in history_results}, {f"call-stop-{i}" for i in range(turns)})
        final = requests[-1]["body"]["messages"]
        # Ten groups may contain more than ten messages when tool pairs are retained.
        self.assertGreater(len(final), 10)
        users = [wire_text(m) for m in final if m["role"] == "user"]
        self.assertLess(len(users), turns)
        self.assertEqual(len(users), len(set(users)))
        self.assertIn('## Context\nComplex command:\n"Stop round 12."', users)
        self.assertNotIn('## Context\nComplex command:\n"Stop round 0."', users)
        tool_results = {m["tool_call_id"]: m["content"] for m in final if m["role"] == "tool"}
        self.assertEqual(tool_results["call-stop-12"], "stopped.")
        self.assertEqual(set(tool_results.values()), {"stopped."})

    async def test_chatbot_entrypoints_select_their_intended_options(self):
        for name in ("Chatbot", "ChatbotWithChatOptions", "ChatbotWithStreaming", "ChatbotWithTools"):
            with self.subTest(sample=name):
                module = load_sample("03.ChatClients", name)
                sdk = async_sdk_context()
                client = Mock()
                with patch.object(module, "create_sdk_client", return_value=sdk) as factory, \
                     patch.object(module, "create_completion_client", return_value=client) as connector, \
                     patch.object(module, "run_chatbot", new_callable=AsyncMock) as loop:
                    await module.main()
                factory.assert_called_once_with("azure")
                connector.assert_called_once_with(sdk, "azure")
                self.assertEqual(loop.await_count, 1)
                self.assertEqual(loop.await_args.args, (client,))
                kwargs = loop.await_args.kwargs
                if name == "Chatbot":
                    self.assertEqual(kwargs, {})
                elif name == "ChatbotWithChatOptions":
                    self.assertEqual(kwargs, {"options": module.chat_options()})
                elif name == "ChatbotWithStreaming":
                    self.assertEqual(kwargs, {"stream": True, "options": {"response_format": chatbot.StepsResponse}})
                else:
                    self.assertEqual(kwargs["system"], chat_clients.TOOLS_SYSTEM)
                    self.assertTrue(kwargs["reduce_history"])
                    self.assertEqual(kwargs["options"]["tool_choice"], "auto")
                    self.assertTrue(kwargs["options"]["allow_multiple_tool_calls"])
                    self.assertEqual({tool.name for tool in kwargs["options"]["tools"]}, {"turn_right", "turn_left", "stop", "forward", "backward"})
                sdk.__aexit__.assert_awaited_once()

    async def test_logging_middleware_logs_only_metadata_and_reraises_errors(self):
        for stream in (False, True):
            for fails in (False, True):
                with self.subTest(stream=stream, fails=fails):
                    error = RuntimeError("sensitive prompt offline-openai-key")
                    result = ChatResponse(messages=[Message("assistant", ["Sensitive answer."])])
                    context = ChatContext(client=Mock(), messages=[Message("system", ["Private system."]), Message("user", ["sensitive prompt"])],
                                          options={"api_key": "offline-openai-key"}, stream=stream)

                    async def proceed():
                        if fails:
                            raise error
                        context.result = result

                    call_next = AsyncMock(side_effect=proceed)
                    with patch.object(chat_middleware, "perf_counter", side_effect=[10.0, 10.125]), self.assertLogs("chat_clients", level="INFO") as logs:
                        if fails:
                            with self.assertRaises(RuntimeError) as raised:
                                await chat_middleware.LoggingChatMiddleware().process(context, call_next)
                            self.assertIs(raised.exception, error)
                        else:
                            await chat_middleware.LoggingChatMiddleware().process(context, call_next)
                            self.assertIs(context.result, result)
                    call_next.assert_awaited_once_with()
                    self.assertEqual(logs.records[0].getMessage(), f"Chat request: 2 messages, streaming={stream}")
                    expected = "Chat request failed after 0.125s" if fails else "Chat response received after 0.125s"
                    self.assertEqual(logs.records[1].getMessage(), expected)
                    self.assertEqual([record.levelname for record in logs.records], ["INFO", "ERROR" if fails else "INFO"])
                    for private in ("Private system.", "sensitive prompt", "offline-openai-key", "Sensitive answer."):
                        self.assertNotIn(private, "\n".join(logs.output))

    async def test_middleware_sample_attaches_logging_to_real_framework(self):
        module = load_sample("03.ChatClients", "ChatClientWithMiddleware")
        sdk, requests = self.configured_sdk(lambda *_: completion_payload("Logged result."))
        with patch.object(module, "create_sdk_client", return_value=sdk), \
             patch.object(module.logging, "basicConfig"), self.assertLogs("chat_clients", level="INFO") as logs:
            await module.main()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
        self.assertIn("Chat request: 1 messages, streaming=False", logs.output[0])
        self.assertIn("Chat response received", logs.output[1])
        self.assertNotIn(chat_clients.ROBOT_PROMPT, "\n".join(logs.output))
        self.assertNotIn(ENV["OPENAI_API_KEY"], "\n".join(logs.output))
        self.assertIn("Assistant: Logged result.", self.output.getvalue())
        self.assertTrue(sdk.is_closed())

    async def test_anthropic_sample_uses_model_max_tokens_and_closes_sdk(self):
        module = load_sample("03.ChatClients", "AnthropicChatClient")
        sdk = async_sdk_context()
        client = Mock(get_response=AsyncMock(return_value=ChatResponse(messages=[Message("assistant", ["Anthropic result."])])))
        with patch.object(module, "AsyncAnthropic", return_value=sdk) as factory, patch.object(module, "AnthropicClient", return_value=client) as connector:
            await module.main()
        factory.assert_called_once_with(api_key="offline-anthropic-key")
        connector.assert_called_once_with(model="claude-offline-model", anthropic_client=sdk)
        messages = client.get_response.await_args.args[0]
        self.assertEqual([(m.role, m.text) for m in messages], [("user", chat_clients.ROBOT_PROMPT)])
        self.assertEqual(client.get_response.await_args.kwargs, {"options": {"max_tokens": 512}})
        self.assertIn("Anthropic result.", self.output.getvalue())
        sdk.__aexit__.assert_awaited_once()

    async def test_ollama_defaults_overrides_and_cleanup_even_on_failure(self):
        module = load_sample("04.ChatClients", "OllamaChatClient")
        for overrides, fails in (({}, False), ({"OLLAMA_HOST": "http://ollama.invalid:11434", "OLLAMA_MODEL": "offline-gemma"}, False), ({}, True)):
            with self.subTest(overrides=overrides, fails=fails), patch.dict(os.environ, overrides):
                sdk = Mock(_client=Mock(aclose=AsyncMock()))
                error = RuntimeError("Offline Ollama failure")
                client = Mock(get_response=AsyncMock(side_effect=error) if fails else AsyncMock(return_value=ChatResponse(messages=[Message("assistant", ["Ollama result."])])))
                with patch.object(module, "AsyncClient", return_value=sdk) as factory, patch.object(module, "OllamaChatClient", return_value=client) as connector:
                    if fails:
                        with self.assertRaises(RuntimeError) as raised:
                            await module.main()
                        self.assertIs(raised.exception, error)
                    else:
                        await module.main()
                factory.assert_called_once_with(host=overrides.get("OLLAMA_HOST", "http://localhost:11434"))
                connector.assert_called_once_with(client=sdk, model=overrides.get("OLLAMA_MODEL", "gemma4:e4b"))
                sdk._client.aclose.assert_awaited_once()
                client.get_response.assert_awaited_once()
                messages = client.get_response.await_args.args[0]
                self.assertEqual([(m.role, m.text) for m in messages[:2]], [("system", module.SYSTEM), ("user", f'Complex command:\n"{chat_clients.TREE_COMMAND}"')])
                if not fails:
                    self.assertEqual(messages[-1].text, "Ollama result.")

    async def test_background_entrypoints_poll_empty_messages_with_response_id(self):
        for name, provider in (("OpenAIResponsesWithBackgroundResponses", "openai"), ("AzureOpenAIResponsesWithBackgroundResponses", "azure")):
            with self.subTest(sample=name):
                module = load_sample("04.ChatClients", name)
                statuses = ("queued", "in_progress", "in_progress", "completed")
                sdk, requests = self.configured_sdk(lambda _, i: responses_payload(statuses[i]), provider=provider)
                client = chat_clients.create_responses_client(sdk, provider)
                framework_calls = self.record_framework_calls(client)
                with patch.object(module, "create_sdk_client", return_value=sdk) as factory, \
                     patch.object(module, "create_responses_client", return_value=client) as connector, \
                     patch.object(background_responses.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                    await module.main()
                factory.assert_called_once_with(*(("azure",) if provider == "azure" else ()))
                connector.assert_called_once_with(*((sdk, "azure") if provider == "azure" else (sdk,)))
                self.assertEqual(len(framework_calls), 4)
                self.assertEqual([(m.role, m.text) for m in framework_calls[0][0]], [("user", chat_clients.BACKGROUND_PROMPT)])
                self.assertEqual(framework_calls[0][1], {"options": {"background": True, "store": True}})
                for messages, kwargs in framework_calls[1:]:
                    self.assertEqual(messages, [])
                    self.assertEqual(kwargs, {"options": {"continuation_token": {"response_id": "resp-offline-job"}}})
                self.assertEqual(sleep.await_args_list, [call(1.0)] * 3)
                self.assertEqual(len(requests), 4)
                prefix = "/openai/v1/responses" if provider == "azure" else "/v1/responses"
                self.assertEqual((requests[0]["method"], requests[0]["path"]), ("POST", prefix))
                self.assertEqual(requests[0]["body"]["model"], chat_clients.model_for(provider, responses=True))
                self.assertTrue(requests[0]["body"]["background"])
                self.assertTrue(requests[0]["body"]["store"])
                for request in requests[1:]:
                    self.assertEqual((request["method"], request["path"], request["body"]), ("GET", prefix + "/resp-offline-job", None))
                self.assertTrue(sdk.is_closed())

    async def test_background_completed_immediately_does_not_poll(self):
        for attempts, delay in ((1, 0), (3, 0.25)):
            with self.subTest(attempts=attempts, delay=delay):
                sdk, requests = self.configured_sdk(lambda *_: responses_payload())
                with patch.object(background_responses.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                    response = await background_responses.run_background_response(
                        chat_clients.create_responses_client(sdk), max_attempts=attempts, delay=delay,
                    )
                self.assertEqual(response.text, "Avoided the tree.")
                self.assertEqual(response.response_id, "resp-offline-job")
                self.assertEqual(response.raw_representation.status, "completed")
                self.assertIsNone(response.continuation_token)
                self.assertEqual(len(requests), 1)
                sleep.assert_not_awaited()
                self.assertNotIn("POLLING", self.output.getvalue())

    async def test_background_terminal_failures_report_status_and_details(self):
        for status, detail in (("failed", "Offline provider failure"), ("cancelled", "None"), ("incomplete", "max_output_tokens")):
            for queued in (False, True):
                with self.subTest(status=status, queued=queued):
                    sdk, requests = self.configured_sdk(lambda _, i: responses_payload("queued" if queued and i == 0 else status))
                    with patch.object(background_responses.asyncio, "sleep", new_callable=AsyncMock) as sleep, self.assertRaises(RuntimeError) as error:
                        await background_responses.run_background_response(chat_clients.create_responses_client(sdk), max_attempts=3, delay=0.1)
                    self.assertIn(f"Background response ended with status {status}:", str(error.exception))
                    self.assertIn(detail, str(error.exception))
                    self.assertEqual(len(requests), 2 if queued else 1)
                    self.assertEqual(sleep.await_count, int(queued))

    async def test_background_timeout_is_bounded_and_identifies_running_response(self):
        sdk, requests = self.configured_sdk(lambda _, i: responses_payload("queued" if i == 0 else "in_progress", response_id="resp-still-running"))
        client = chat_clients.create_responses_client(sdk)
        framework_calls = self.record_framework_calls(client)
        with patch.object(background_responses.asyncio, "sleep", new_callable=AsyncMock) as sleep, self.assertRaises(TimeoutError) as error:
            await background_responses.run_background_response(client, max_attempts=3, delay=0.25)
        self.assertEqual(str(error.exception), "Polling timeout after 3 attempts. Response resp-still-running is still running; retrieve or cancel it with the provider's Responses API.")
        self.assertEqual(sleep.await_args_list, [call(0.25)] * 3)
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(framework_calls), 4)
        for messages, kwargs in framework_calls[1:]:
            self.assertEqual(messages, [])
            self.assertEqual(kwargs, {"options": {"continuation_token": {"response_id": "resp-still-running"}}})
        self.assertEqual([request["method"] for request in requests], ["POST", "GET", "GET", "GET"])
        self.assertTrue(all(request["path"].endswith("/resp-still-running") for request in requests[1:]))

    async def test_background_invalid_bounds_and_provider_errors_propagate(self):
        for kwargs in ({"max_attempts": 0}, {"max_attempts": -2}, {"delay": -0.1}):
            with self.subTest(kwargs=kwargs):
                client = Mock(get_response=AsyncMock())
                with self.assertRaisesRegex(ValueError, "max_attempts must be positive and delay nonnegative"):
                    await background_responses.run_background_response(client, **kwargs)
                client.get_response.assert_not_called()
        for queued in (False, True):
            with self.subTest(queued=queued):
                sdk, requests = self.configured_sdk(lambda _, i: responses_payload("queued") if queued and i == 0 else httpx.Response(
                    503, json={"error": {"message": "offline service unavailable", "type": "server_error", "code": "server_error"}},
                ))
                with patch.object(background_responses.asyncio, "sleep", new_callable=AsyncMock), self.assertRaises(ChatClientException) as error:
                    await background_responses.run_background_response(chat_clients.create_responses_client(sdk), max_attempts=3, delay=0)
                self.assertIn("offline service unavailable", str(error.exception))
                self.assertEqual(len(requests), 2 if queued else 1)


if __name__ == "__main__":
    unittest.main()
