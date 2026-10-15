"""Offline ONNX adapter tests: real Agent Framework, fake native GenAI API."""

import asyncio
import gc
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
import weakref
from collections.abc import Awaitable
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_framework import Content, Message, ResponseStream

from shared.onnx_client import OnnxChatClient, stream_with_timings

ROOT = Path(__file__).resolve().parents[1]


def fake_runtime():
    """Mimic the 0.12.0 bindings, notably Generator.set_inputs and Tensor.shape()."""
    state = SimpleNamespace(
        calls=[], prompts=[], options=[], handles=[], tokens=[10, 11, 12],
        prompt_length=7, image_prompt_length=23, template_error=False,
        processor_error=False, generation_error=False, load_error=False,
        cuda_available=True, dml_available=True, image_bytes=[],
    )

    class Native:
        def __init__(self, name):
            state.handles.append((name, weakref.ref(self)))

    class Config(Native):
        def __init__(self, path):
            super().__init__("config")
            state.calls.append(("config", path))

        def clear_providers(self):
            state.calls.append(("clear_providers",))

        def append_provider(self, provider):
            state.calls.append(("append_provider", provider))

        def set_provider_option(self, provider, option, value):
            state.calls.append(("provider_option", provider, option, value))

    class Model(Native):
        def __init__(self, config):
            super().__init__("model")
            if state.load_error:
                raise RuntimeError("model load failed")
            self.config = config

        def create_multimodal_processor(self):
            return Processor(self)

    class Tokenizer(Native):
        def __init__(self, model):
            super().__init__("tokenizer")
            self.model = model

        def apply_chat_template(self, messages, *, template_str=None, tools=None, add_generation_prompt=True):
            turns = json.loads(messages)
            state.calls.append(("chat_template", turns, template_str, add_generation_prompt))
            if state.template_error:
                raise RuntimeError("No embedded chat template")
            return "NATIVE:" + "|".join(f"{t['role']}:{t['content']}" for t in turns) + ":ASSISTANT"

        def encode(self, prompt):
            state.prompts.append(prompt)
            return list(range(state.prompt_length))

        def create_stream(self):
            return Decoder()

    class Decoder(Native):
        def __init__(self):
            super().__init__("decoder")

        def decode(self, token):
            return {10: "forward", 11: "", 12: ", stop"}[token]

    class Images(Native):
        def __init__(self, images):
            super().__init__("images")
            self.images = images

        @staticmethod
        def open_bytes(*images):
            state.image_bytes.append(images)
            return Images(images)

    class Tensor(Native):
        def __init__(self):
            super().__init__("tensor")

        def shape(self):
            return [1, state.image_prompt_length]

    class NamedTensors(Native):
        def __init__(self):
            super().__init__("inputs")
            self.tensor = Tensor()

        def __getitem__(self, name):
            if name != "input_ids":
                raise KeyError(name)
            return self.tensor

    class Processor(Native):
        def __init__(self, model):
            super().__init__("processor")
            self.model = model

        def __call__(self, prompt, *, images=None):
            state.prompts.append(prompt)
            if state.processor_error:
                raise RuntimeError("bad vision assets")
            state.calls.append(("process_images", images.images if images else None))
            return NamedTensors()

    class Params(Native):
        def __init__(self, model):
            super().__init__("params")
            self.model = model
            self.search = {}

        def set_search_options(self, **options):
            self.search.update(options)
            state.options.append(options)

    class Generator(Native):
        def __init__(self, model, params):
            super().__init__("generator")
            self.model = model
            self.params = params
            self.index = 0
            self.sequence_length = 0
            self.inputs = None

        def set_inputs(self, inputs):
            self.inputs = inputs
            self.sequence_length = inputs["input_ids"].shape()[1]
            state.calls.append(("set_inputs", self.sequence_length))

        def append_tokens(self, tokens):
            self.sequence_length = len(tokens)
            state.calls.append(("append_tokens", list(tokens)))

        def is_done(self):
            return self.index == len(state.tokens) or self.sequence_length >= self.params.search["max_length"]

        def generate_next_token(self):
            if state.generation_error:
                raise RuntimeError("native generation failed")
            self.index += 1
            self.sequence_length += 1

        def get_next_tokens(self):
            return [state.tokens[self.index - 1]]

    module = SimpleNamespace(
        __version__="0.12.0", Config=Config, Model=Model, Tokenizer=Tokenizer,
        Images=Images, GeneratorParams=Params, Generator=Generator,
        is_cuda_available=lambda: state.cuda_available,
        is_dml_available=lambda: state.dml_available,
    )
    return module, state


class OnnxClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model_path = Path(self.temporary.name)
        self.write_model()
        self.runtime, self.native = fake_runtime()
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        env_path = patch("shared.onnx_client._ROOT_ENV", self.model_path / ".env")
        env_path.start()
        self.addCleanup(env_path.stop)
        runtime_patch = patch.dict(sys.modules, {"onnxruntime_genai": self.runtime})
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)

    def write_model(self, model_type="phi3", *, vision=False, context=1000):
        model = {
            "type": model_type, "context_length": context,
            "decoder": {"filename": "model.onnx"},
        }
        (self.model_path / "model.onnx").write_bytes(b"fake model")
        (self.model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        if vision:
            model["vision"] = {"filename": "vision.onnx", "config_filename": "processor_config.json"}
            model["embedding"] = {"filename": "embedding.onnx"}
            for name in ["vision.onnx", "embedding.onnx", "processor_config.json"]:
                (self.model_path / name).write_bytes(b"fake asset")
        (self.model_path / "genai_config.json").write_text(json.dumps({"model": model}), encoding="utf-8")

    def client(self, **kwargs):
        client = OnnxChatClient(self.model_path, **kwargs)
        self.addCleanup(client.close)
        return client

    def assert_released(self, *names):
        gc.collect()
        live = [name for name, reference in self.native.handles if reference() is not None and name in names]
        self.assertEqual(live, [], f"Native objects still retained: {live}")

    async def consume(self, client, messages=None, options=None):
        stream = client.get_response(
            messages if messages is not None else [Message("user", ["Go."])],
            stream=True, options=options,
        )
        self.assertIsInstance(stream, ResponseStream)
        updates = [update async for update in stream]
        return updates, await stream.get_final_response()

    def test_constructor_loads_explicit_root_dotenv_without_overriding(self):
        with patch("shared.onnx_client._ROOT_ENV", ROOT / ".env"):
            with patch("shared.onnx_client.load_dotenv", return_value=False) as loader:
                self.client()
        loader.assert_called_once_with(dotenv_path=ROOT / ".env", override=False)
        self.assertEqual(self.native.calls, [])

    def test_dotenv_model_provider_and_template_respect_overrides(self):
        (self.model_path / ".env").write_text(
            f"ONNX_MODEL_PATH='{self.model_path}'\n"
            f"ONNX_MULTIMODAL_MODEL_PATH='{self.model_path / 'vision'}'\n"
            'ONNX_PROVIDER=cuda\nONNX_CHAT_TEMPLATE="[INST] {prompt} [/INST]"\n',
            encoding="utf-8",
        )
        client = OnnxChatClient()
        self.addCleanup(client.close)
        self.assertEqual(client.model_path, self.model_path)
        self.assertEqual(client.provider, "cuda")
        self.assertEqual(client.chat_template, "[INST] {prompt} [/INST]")
        os.environ["ONNX_MODEL_PATH"] = str(self.model_path / "environment")
        os.environ["ONNX_PROVIDER"] = "dml"
        environment_client = OnnxChatClient()
        self.addCleanup(environment_client.close)
        self.assertEqual(environment_client.model_path, self.model_path / "environment")
        self.assertEqual(environment_client.provider, "dml")
        explicit = self.client(provider="cpu", chat_template="{prompt}")
        self.assertEqual(explicit.model_path, self.model_path)
        self.assertEqual(explicit.provider, "cpu")
        self.assertEqual(explicit.chat_template, "{prompt}")
        vision_client = OnnxChatClient(vision=True)
        self.addCleanup(vision_client.close)
        self.assertEqual(vision_client.model_path, self.model_path / "vision")

    def test_model_path_errors_identify_environment_and_root_dotenv(self):
        for vision, key in [(False, "ONNX_MODEL_PATH"), (True, "ONNX_MULTIMODAL_MODEL_PATH")]:
            with self.subTest(vision=vision):
                os.environ[key] = "  "
                with self.assertRaisesRegex(RuntimeError, f"Set {key} in the root .env"):
                    OnnxChatClient(vision=vision).load()
                os.environ[key] = str(self.model_path / "missing")
                with self.assertRaisesRegex(RuntimeError, f"{key} points to a missing model folder"):
                    OnnxChatClient(vision=vision).load()
        with self.assertRaisesRegex(RuntimeError, "not an .onnx file"):
            OnnxChatClient(self.model_path / "model.onnx").load()
        self.assertEqual(self.native.calls, [])

    def test_constructor_is_lazy_and_load_is_idempotent(self):
        client = self.client()
        with patch("shared.onnx_client.importlib.import_module") as importer:
            self.assertEqual(self.native.calls, [])
            importer.assert_not_called()
        client.load()
        client.load()
        self.assertEqual(self.native.calls, [("config", str(self.model_path)), ("clear_providers",)])
        self.assertGreaterEqual(client.load_seconds, 0)
        client.close()
        self.assert_released("model", "config", "tokenizer")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.load()

    async def test_native_template_stream_and_final_usage_count_actual_tokens(self):
        client = self.client()
        updates, response = await self.consume(client, options={"max_tokens": 512})
        self.assertEqual(self.native.prompts, ["NATIVE:user:Go.:ASSISTANT"])
        self.assertEqual(self.native.calls[2], ("chat_template", [{"role": "user", "content": "Go."}], "", True))
        self.assertEqual([update.text for update in updates], ["forward", "", ", stop", ""])
        self.assertEqual(response.text, "forward, stop")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage_details, {"input_token_count": 7, "output_token_count": 3, "total_token_count": 10})
        self.assertEqual(len({update.response_id for update in updates}), 1)
        self.assertEqual(self.native.options[0]["max_length"], 519)
        self.assertEqual(client.last_stats.output_token_count, 3)
        self.assert_released("generator", "params", "decoder")
        self.assertIsNotNone(client._model)

    async def test_nonstream_and_multiturn_instructions(self):
        client = self.client()
        pending = client.get_response(
            [Message("user", ["left"]), Message("assistant", ["turn left"]), Message("user", ["stop"])],
            options={"instructions": "Only moves", "max_tokens": 10},
        )
        self.assertIsInstance(pending, Awaitable)
        self.assertNotIsInstance(pending, ResponseStream)
        response = await pending
        self.assertEqual(response.text, "forward, stop")
        self.assertEqual(self.native.prompts, [
            "NATIVE:system:Only moves|user:left|assistant:turn left|user:stop:ASSISTANT"
        ])
        self.assertEqual(response.usage_details["output_token_count"], 3)
        self.assert_released("generator", "params", "decoder")

    async def test_phi_fallback_uses_roles_and_generation_prefix(self):
        self.native.template_error = True
        client = self.client()
        await self.consume(client, [Message("system", ["moves"]), Message("user", ["Go."])])
        self.assertEqual(self.native.prompts, ["<|system|>\nmoves<|end|>\n<|user|>\nGo.<|end|>\n<|assistant|>\n"])

    async def test_missing_template_method_uses_phi_fallback(self):
        del self.runtime.Tokenizer.apply_chat_template
        await self.consume(self.client())
        self.assertEqual(self.native.prompts, ["<|user|>\nGo.<|end|>\n<|assistant|>\n"])

    async def test_mistral_custom_template_overrides_native(self):
        self.write_model("mistral")
        os.environ["ONNX_CHAT_TEMPLATE"] = "[INST] {prompt} [/INST] {{json}}"
        await self.consume(self.client())
        self.assertEqual(self.native.prompts, ["[INST] Go. [/INST] {json}"])
        self.assertFalse(any(call[0] == "chat_template" for call in self.native.calls))

    async def test_nonphi_does_not_silently_assume_phi_template(self):
        self.write_model("mistral")
        self.native.template_error = True
        with self.assertRaisesRegex(RuntimeError, "ONNX_CHAT_TEMPLATE"):
            await self.consume(self.client())
        self.assertEqual(self.native.prompts, [])
        self.assert_released("generator", "params")

    async def test_local_jinja_passed_to_native_and_failures_not_ignored(self):
        (self.model_path / "chat_template.jinja").write_text("a model-specific Jinja template", encoding="utf-8")
        await self.consume(self.client())
        self.assertEqual(self.native.calls[2][2], "a model-specific Jinja template")
        self.native.template_error = True
        with self.assertRaisesRegex(RuntimeError, "chat_template.jinja"):
            await self.consume(self.client())

    async def test_custom_template_validation(self):
        for template in ["missing placeholder", "{prompt} {unknown}", "{prompt} {"]:
            with self.subTest(template=template):
                with self.assertRaisesRegex(ValueError, "ONNX_CHAT_TEMPLATE"):
                    await self.consume(self.client(chat_template=template))
        with self.assertRaisesRegex(ValueError, "single user"):
            await self.consume(self.client(chat_template="{prompt}"), [Message("system", ["moves"])])

    async def test_output_budget_and_context_cap_are_not_confused(self):
        for context, budget, expected_length, count in [(1000, 2, 9, 2), (9, 512, 9, 2), (10, 512, 10, 3)]:
            with self.subTest(context=context, budget=budget):
                self.write_model(context=context)
                client = self.client()
                updates, response = await self.consume(client, options={"max_tokens": budget})
                self.assertEqual(self.native.options[-1]["max_length"], expected_length)
                self.assertEqual(client.last_stats.max_output_tokens, count)
                self.assertEqual(response.usage_details["output_token_count"], count)
                self.assertEqual(response.finish_reason, "length")
                self.assertEqual(len(updates), count + 1)

    async def test_prompt_filling_context_rejected(self):
        for context in [6, 7]:
            with self.subTest(context=context):
                self.write_model(context=context)
                with self.assertRaisesRegex(ValueError, "no room"):
                    await self.consume(self.client())
        self.assertEqual(self.native.options, [])
        self.assert_released("generator", "params")

    async def test_max_tokens_validation_and_unsupported_options(self):
        for budget in [0, -1, True, 2.5]:
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "max_tokens"):
                    await self.consume(self.client(), options={"max_tokens": budget})
        for options in [{"tools": []}, {"stop": ["stop"]}, {"response_format": {"type": "json_object"}}]:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "Unsupported ONNX"):
                    await self.consume(self.client(), options=options)
        with self.assertRaisesRegex(ValueError, "Unsupported ONNX"):
            await self.client().get_response([Message("user", ["Go."])], client_kwargs={"extra": True})

    async def test_supported_search_options(self):
        await self.consume(self.client(), options={"max_tokens": 12, "temperature": 0.7, "top_p": 0.8, "seed": 42})
        self.assertEqual(self.native.options[0], {
            "max_length": 19, "batch_size": 1, "num_beams": 1, "do_sample": True,
            "temperature": 0.7, "top_p": 0.8, "random_seed": 42,
        })
        await self.consume(self.client(), options={"temperature": 0, "top_p": 1})
        self.assertFalse(self.native.options[-1]["do_sample"])
        self.assertEqual(self.native.options[-1]["top_p"], 1)

    async def test_invalid_messages_are_rejected_before_generation(self):
        self.write_model("phi3v", vision=True)
        for messages in [[], [Message("tool", ["result"])], [Message("assistant", [Content.from_data(b"jpeg", "image/jpeg")])]]:
            with self.subTest(messages=messages):
                with self.assertRaisesRegex(ValueError, "At least one|not tool calls|user messages"):
                    await self.consume(self.client(vision=True), messages)
        self.assertEqual(self.native.prompts, [])
        self.assertEqual(self.native.options, [])

    def test_cpu_and_opt_in_provider_configuration(self):
        for provider in ["cpu", "cuda", "dml"]:
            with self.subTest(provider=provider):
                self.native.calls.clear()
                os.environ["ONNX_PROVIDER"] = provider
                client = self.client()
                client.load()
                expected = [("config", str(self.model_path)), ("clear_providers",)]
                if provider != "cpu":
                    expected.append(("append_provider", provider))
                if provider == "cuda":
                    expected.extend([
                        ("provider_option", "cuda", "device_id", "0"),
                        ("provider_option", "cuda", "enable_cuda_graph", "0"),
                    ])
                self.assertEqual(self.native.calls, expected)
                client.close()
        with self.assertRaisesRegex(ValueError, "cpu, cuda or dml"):
            self.client(provider="unknown")

    def test_cuda_unavailable_fails_without_cpu_fallback(self):
        self.native.cuda_available = False
        with self.assertRaisesRegex(RuntimeError, "does not support cuda"):
            self.client(provider="cuda").load()
        self.assertEqual(self.native.calls, [])

    def test_environment_paths_are_distinct_and_explicit_overrides_win(self):
        os.environ.update(ONNX_MODEL_PATH=str(self.model_path), ONNX_MULTIMODAL_MODEL_PATH=str(self.model_path / "vision"), ONNX_PROVIDER="cuda")
        client = OnnxChatClient()
        self.addCleanup(client.close)
        vision = OnnxChatClient(vision=True)
        self.addCleanup(vision.close)
        self.assertEqual(client.model_path, self.model_path)
        self.assertEqual(vision.model_path, self.model_path / "vision")
        self.assertEqual(self.client(provider="cpu").provider, "cpu")
        os.environ.pop("ONNX_MULTIMODAL_MODEL_PATH")
        with self.assertRaisesRegex(RuntimeError, "ONNX_MULTIMODAL_MODEL_PATH"):
            OnnxChatClient(vision=True).load()

    def test_missing_runtime_old_version_and_native_load_errors_have_guidance(self):
        for error in [ModuleNotFoundError("not installed"), OSError("missing DLL")]:
            with self.subTest(error=error):
                with patch("shared.onnx_client.importlib.import_module", side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, "requirements-onnx.txt"):
                        self.client().load()
        self.runtime.__version__ = "0.11.0"
        with self.assertRaisesRegex(RuntimeError, "0.12.0 or newer"):
            self.client().load()
        self.runtime.__version__ = "0.12.0"
        self.native.load_error = True
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, "model load failed.*requirements-onnx.txt"):
            client.load()
        self.assertIsNone(client._model)
        self.assertIsNone(client._runtime)

    def test_missing_path_config_and_assets_report_clear_errors(self):
        with self.assertRaisesRegex(RuntimeError, "Set ONNX_MODEL_PATH"):
            OnnxChatClient().load()
        config = self.model_path / "genai_config.json"
        original = config.read_text(encoding="utf-8")
        config.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Invalid or missing ONNX configuration"):
            self.client().load()
        config.write_text(original, encoding="utf-8")
        (self.model_path / "model.onnx").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing assets: model.onnx"):
            self.client().load()
        self.assertEqual(self.native.calls, [])

    def test_malformed_model_metadata_reports_configuration_errors(self):
        for config in [[], {"model": []}, {"model": {"type": [], "context_length": 100}},
                       {"model": {"type": "phi3", "context_length": True}},
                       {"model": {"type": "phi3", "context_length": 0}}]:
            with self.subTest(config=config):
                (self.model_path / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Invalid or missing ONNX configuration"):
                    self.client().load()
        self.write_model("phi3v", vision=True)
        config_path = self.model_path / "genai_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        del config["model"]["vision"]["filename"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "missing model.vision.filename"):
            self.client(vision=True).load()
        self.assertEqual(self.native.calls, [])

    async def test_vision_processes_bytes_markers_and_expanded_prompt_length(self):
        for model_type in ["phi3v", "phi4mm"]:
            with self.subTest(model_type=model_type):
                self.write_model(model_type, vision=True)
                self.native.prompts.clear()
                client = self.client(vision=True)
                messages = [Message("user", [
                    Content.from_text("Look at the image and tell what you see."),
                    Content.from_data(b"jpeg one", "image/jpeg"),
                    Content.from_data(b"jpeg two", "image/jpeg"),
                ])]
                _, response = await self.consume(client, messages, {"max_tokens": 512})
                self.assertEqual(self.native.image_bytes[-1], (b"jpeg one", b"jpeg two"))
                self.assertEqual(self.native.prompts, [
                    "NATIVE:user:<|image_1|>\n<|image_2|>\nLook at the image and tell what you see.:ASSISTANT"
                ])
                self.assertIn(("set_inputs", 23), self.native.calls)
                self.assertEqual(self.native.options[-1]["max_length"], 535)
                self.assertEqual(response.usage_details, {"input_token_count": 23, "output_token_count": 3, "total_token_count": 26})
                self.assert_released("generator", "params", "images", "inputs", "tensor", "decoder")
                client.close()
                self.assert_released("processor", "model", "config", "tokenizer")

    async def test_vision_phi_fallback_preserves_image_marker(self):
        self.write_model("phi3v", vision=True)
        self.native.template_error = True
        await self.consume(self.client(vision=True), [Message("user", [
            Content.from_text("Describe."), Content.from_data(b"jpeg", "image/jpeg")
        ])])
        self.assertEqual(self.native.prompts, ["<|user|>\n<|image_1|>\nDescribe.<|end|>\n<|assistant|>\n"])

    def test_text_only_model_and_incomplete_vision_assets_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "text-only phi-4-mini-instruct"):
            self.client(vision=True).load()
        self.write_model("phi3v", vision=True)
        (self.model_path / "processor_config.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing assets: processor_config.json"):
            self.client(vision=True).load()
        self.assertEqual(self.native.calls, [])

    async def test_text_client_refuses_images_and_other_content_is_not_ignored(self):
        for content, error in [
            (Content.from_data(b"jpeg", "image/jpeg"), RuntimeError),
            (Content.from_data(b"audio", "audio/mpeg"), ValueError),
            (Content.from_uri("https://example.com/image.jpg", media_type="image/jpeg"), ValueError),
            (Content.from_function_call("call", "move", arguments={}), ValueError),
        ]:
            with self.subTest(content=content.type):
                with self.assertRaises(error):
                    await self.consume(self.client(), [Message("user", [Content.from_text("Go."), content])])
        self.assertEqual(self.native.prompts, [])
        self.assertEqual(self.native.options, [])

    async def test_invalid_vision_bytes_processing_and_dropped_markers_fail(self):
        self.write_model("phi3v", vision=True)
        for image in [Content.from_data(b"", "image/jpeg"), Content("data", uri="data:image/jpeg;base64,??", media_type="image/jpeg")]:
            with self.subTest(uri=image.uri):
                with self.assertRaisesRegex(ValueError, "Image data is empty|Invalid base64"):
                    await self.consume(self.client(vision=True), [Message("user", [image])])
        self.native.processor_error = True
        with self.assertRaisesRegex(RuntimeError, "Could not process image input.*bad vision assets"):
            await self.consume(self.client(vision=True), [Message("user", [Content.from_data(b"jpeg", "image/jpeg")])])
        self.assert_released("images", "inputs", "generator", "params")
        self.native.processor_error = False
        with patch.object(self.runtime.Tokenizer, "apply_chat_template", return_value="No image here"):
            with self.assertRaisesRegex(RuntimeError, "dropped a Phi vision image marker"):
                await self.consume(self.client(vision=True), [Message("user", [Content.from_data(b"jpeg", "image/jpeg")])])

    async def test_vision_budget_caps_processed_not_text_only_prompt(self):
        self.write_model("phi3v", vision=True, context=25)
        client = self.client(vision=True)
        _, response = await self.consume(client, [Message("user", [Content.from_data(b"jpeg", "image/jpeg")])])
        self.assertEqual(self.native.options[0]["max_length"], 25)
        self.assertEqual(client.last_stats.max_output_tokens, 2)
        self.assertEqual(response.usage_details["input_token_count"], 23)
        self.assertEqual(response.usage_details["output_token_count"], 2)

    async def test_cleanup_on_generation_error_and_cancelled_stream(self):
        client = self.client()
        self.native.generation_error = True
        with self.assertRaisesRegex(RuntimeError, "native generation failed"):
            await self.consume(client)
        self.assert_released("generator", "params", "decoder")
        self.native.generation_error = False
        stream = client.get_response([Message("user", ["Go."])], stream=True)
        self.assertEqual((await stream.__anext__()).text, "forward")
        task = asyncio.create_task(stream.__anext__())
        asyncio.get_running_loop().call_soon(task.cancel)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_released("generator", "params", "decoder")
        self.assertIsNone(client._active)
        _, response = await self.consume(client)
        self.assertEqual(response.text, "forward, stop")

    async def test_close_releases_abandoned_stream_and_blocks_concurrent_request(self):
        client = self.client()
        stream = client.get_response([Message("user", ["Go."])], stream=True)
        await stream.__anext__()
        with self.assertRaisesRegex(RuntimeError, "one active generation"):
            await self.consume(client)
        client.close()
        client.close()
        self.assert_released("generator", "params", "decoder", "model", "tokenizer", "config")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await stream.__anext__()

    async def test_context_manager_releases_resources(self):
        with self.client() as client:
            await self.consume(client)
        self.assert_released("model", "config", "tokenizer", "generator", "params", "decoder")

    async def test_zero_and_single_token_timing_no_divide_by_zero(self):
        for tokens, count in [([], 0), ([10], 1)]:
            with self.subTest(tokens=tokens):
                self.native.tokens = tokens
                client = self.client()
                output = io.StringIO()
                with redirect_stdout(output):
                    stats = await stream_with_timings(client, [Message("user", ["Go."])])
                self.assertEqual(stats.output_token_count, count)
                self.assertEqual(stats.generation_seconds, 0)
                self.assertEqual(stats.tokens_per_second, 0)
                self.assertIn(f"({count} tokens in 0 ms)", output.getvalue())
                if count == 0:
                    self.assertIn("n/a (no output tokens)", output.getvalue())
                else:
                    self.assertIn("First token time:", output.getvalue())
                    self.assertIn("Last token time:", output.getvalue())

    async def test_timing_uses_native_timestamps_not_usage_chunks(self):
        client = self.client()
        client.load()
        output = io.StringIO()
        with patch("shared.onnx_client.perf_counter", side_effect=[100.0, 101.0, 102.0, 102.5, 103.0]):
            with redirect_stdout(output):
                stats = await stream_with_timings(client, [Message("user", ["Go."])])
        self.assertEqual(stats.output_token_count, 3)
        self.assertEqual(stats.first_token_at, 102.0)
        self.assertEqual(stats.last_token_at, 103.0)
        self.assertAlmostEqual(stats.generation_seconds, 1.0)
        self.assertAlmostEqual(stats.tokens_per_second, 3.0)
        self.assertIn("First token time: 2000 ms", output.getvalue())
        self.assertIn("Last token time: 3000 ms", output.getvalue())
        self.assertIn("3 tokens/second (3 tokens in 1000 ms)", output.getvalue())

    async def test_samples_preserve_prompts_paths_and_512_output_budget(self):
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        text = runpy.run_path(str(ROOT / "04.ChatClients" / "OnnxChatClient" / "main.py"))
        vision = runpy.run_path(str(ROOT / "04.ChatClients" / "OnnxChatClientMultiModal" / "main.py"))
        self.assertIn("forward, backward, turn left, turn right, and stop", text["PROMPT"])
        self.assertIn('"There is a tree directly in front of the car. Avoid it and then come back to the original path."', text["PROMPT"])
        with patch.dict(text["main"].__globals__, {"OnnxChatClient": lambda: self.client(), "stream_with_timings": AsyncMock()}) as globals_:
            with redirect_stdout(io.StringIO()):
                await text["main"]()
            args = globals_["stream_with_timings"].await_args
            self.assertEqual(args.args[1][0].text, text["PROMPT"])
            self.assertEqual(args.kwargs, {"max_output_tokens": 512})
            self.assertTrue(args.args[0]._closed)
        jpeg = self.model_path / "path.jpg"
        jpeg.write_bytes(b"local JPEG bytes")
        os.environ["ONNX_IMAGE_PATH"] = str(jpeg)
        with patch.dict(vision["main"].__globals__, {"OnnxChatClient": lambda **kw: self.client(**kw), "stream_with_timings": AsyncMock()}) as globals_:
            await vision["main"]()
            args = globals_["stream_with_timings"].await_args
            message = args.args[1][0]
            self.assertEqual(message.text, "Look at the image and tell what you see.")
            self.assertEqual(message.contents[1].media_type, "image/jpeg")
            self.assertEqual(message.contents[1].uri, "data:image/jpeg;base64,bG9jYWwgSlBFRyBieXRlcw==")
            self.assertEqual(args.kwargs, {"max_output_tokens": 512})
            self.assertTrue(args.args[0].vision)
            self.assertTrue(args.args[0]._closed)
        os.environ.pop("ONNX_IMAGE_PATH")
        (self.model_path / ".env").write_text(f"ONNX_IMAGE_PATH='{jpeg}'\n", encoding="utf-8")
        with patch.dict(vision["main"].__globals__, {"OnnxChatClient": lambda **kw: self.client(**kw), "stream_with_timings": AsyncMock()}) as globals_:
            await vision["main"]()
            message = globals_["stream_with_timings"].await_args.args[1][0]
            self.assertEqual(message.contents[1].uri, "data:image/jpeg;base64,bG9jYWwgSlBFRyBieXRlcw==")
        (self.model_path / ".env").unlink()
        os.environ["ONNX_IMAGE_PATH"] = str(self.model_path / "missing.jpg")
        with self.assertRaisesRegex(RuntimeError, "JPEG input not found.*ONNX_IMAGE_PATH"):
            await vision["main"]()
        os.environ.pop("ONNX_IMAGE_PATH")
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(RuntimeError, r"OnnxChatClientMultiModal\\Data\\path\.jpg"):
                await vision["main"]()


class InstalledOnnxApiTests(unittest.TestCase):
    def test_optional_native_api_matches_adapter_without_loading_model(self):
        try:
            import onnxruntime_genai as runtime
        except ModuleNotFoundError as exc:
            if exc.name != "onnxruntime_genai":
                raise
            self.skipTest("Optional ONNX Runtime GenAI is not installed")

        required = {
            "Config": ["clear_providers", "append_provider", "set_provider_option"],
            "Model": ["create_multimodal_processor"],
            "Tokenizer": ["apply_chat_template", "encode", "create_stream"],
            "TokenizerStream": ["decode"],
            "GeneratorParams": ["set_search_options"],
            "Generator": ["set_inputs", "append_tokens", "is_done", "generate_next_token", "get_next_tokens"],
            "Images": ["open_bytes"],
            "MultiModalProcessor": ["__call__"],
            "Tensor": ["shape"],
            "NamedTensors": ["__getitem__"],
        }
        for name, methods in required.items():
            cls = getattr(runtime, name)
            for method in methods:
                with self.subTest(native_method=f"{name}.{method}"):
                    self.assertTrue(callable(getattr(cls, method, None)))
        # Pybind methods do not expose inspect.signature; their docs give the
        # actual installed signatures. No native constructors/inference run.
        template_doc = runtime.Tokenizer.apply_chat_template.__doc__
        self.assertIn("messages: str", template_doc)
        self.assertIn("*, template_str: str", template_doc)
        self.assertIn("add_generation_prompt: bool", template_doc)
        self.assertIn("NamedTensors", runtime.Generator.set_inputs.__doc__)
        self.assertIn("arg2: str", runtime.Config.set_provider_option.__doc__)
        self.assertIn("list[int]", runtime.Tensor.shape.__doc__)


if __name__ == "__main__":
    unittest.main()
