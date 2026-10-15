"""Local ONNX Runtime GenAI adapter for Agent Framework core 1.8.1.

Requires onnxruntime-genai >= 0.12.0 (CPU), or the matching
onnxruntime-genai-cuda / onnxruntime-genai-directml distribution, NOT all three.
Install the parent's optional requirements-onnx.txt; no models are downloaded.

ONNX_MODEL_PATH / ONNX_MULTIMODAL_MODEL_PATH must name complete exported model
folders, including genai_config.json, tokenizer.json and referenced ONNX/assets.
Vision is deliberately limited to Phi-3/3.5 vision (phi3v) and Phi-4 multimodal
(phi4mm), with their vision components and processor_config.json.
The repository root .env is loaded explicitly when constructing a client,
without overriding existing environment variables or explicit constructor args.
Use single quotes around Windows paths in .env to preserve backslashes.
ONNX_PROVIDER defaults to cpu; cuda uses device 0 with enable_cuda_graph=0.

The model's Tokenizer.apply_chat_template and optional chat_template.jinja are
used by default. ONNX_CHAT_TEMPLATE overrides this with a Python format string
containing {prompt}, for a single user turn (escape literal braces as {{ / }}).
Without a usable native template, only Phi-3/4 role delimiters are assumed:
<|user|>\\n{prompt}<|end|>\\n<|assistant|>\\n. Phi vision inserts <|image_1|> etc.
Other model families, including Mistral, require their own correct template.

Native API references (0.12.0 bindings take precedence over preview docs):
https://github.com/microsoft/onnxruntime-genai/blob/rel-0.12.0/examples/python/model-qa.py
https://github.com/microsoft/onnxruntime-genai/blob/rel-0.12.0/examples/python/model-mm.py
https://onnxruntime.ai/docs/genai/api/python.html

Inference is synchronous native work between async streaming updates; one
request at a time per client. Use a context manager or close() to release native
resources, including when abandoning a stream before it finishes.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from agent_framework import (
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from dotenv import load_dotenv

_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
MIN_GENAI_VERSION = "0.12.0"
_INSTALL_GUIDANCE = (
    "Install the optional dependencies with python -m pip install -r requirements-onnx.txt "
    "(onnxruntime-genai>=0.12.0 for CPU; use only the matching CUDA or DirectML "
    "distribution for those providers). Check native DLLs and provider prerequisites."
)
_PHI_TYPES = {"phi3", "phi3small", "phi3v", "phi4mm"}
_VISION_TYPES = {"phi3v", "phi4mm"}


@dataclass
class GenerationStats:
    """Native token counts and monotonic timestamps, including empty/EOS tokens."""

    started_at: float = field(default_factory=perf_counter)
    input_token_count: int = 0
    output_token_count: int = 0
    max_output_tokens: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None

    @property
    def generation_seconds(self) -> float:
        if self.first_token_at is None or self.last_token_at is None:
            return 0.0
        return max(0.0, self.last_token_at - self.first_token_at)

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.generation_seconds
        return self.output_token_count / elapsed if elapsed > 0 else 0.0


@dataclass
class _Generation:
    generator: Any = None
    params: Any = None
    images: Any = None
    inputs: Any = None
    decoder: Any = None

    def close(self) -> None:
        # GenAI's pybind objects release native handles when references are dropped.
        # Release the generator first, including its captured graph and KV cache.
        self.generator = None
        self.params = None
        self.inputs = None
        self.images = None
        self.decoder = None


class OnnxChatClient(BaseChatClient):
    """Small text/Phi-vision client; max_tokens is an OUTPUT-token budget."""

    OTEL_PROVIDER_NAME = "onnxruntime-genai"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        provider: str | None = None,
        vision: bool = False,
        chat_template: str | None = None,
    ) -> None:
        super().__init__()
        load_dotenv(dotenv_path=_ROOT_ENV, override=False)
        self.vision = vision
        self._path_env = "ONNX_MULTIMODAL_MODEL_PATH" if vision else "ONNX_MODEL_PATH"
        configured_path = model_path if model_path is not None else os.getenv(self._path_env)
        if isinstance(configured_path, str):
            configured_path = configured_path.strip()
        self.model_path = Path(configured_path).expanduser() if configured_path else None
        self.provider = (provider if provider is not None else os.getenv("ONNX_PROVIDER", "cpu")).lower()
        if self.provider not in {"cpu", "cuda", "dml"}:
            raise ValueError("ONNX_PROVIDER must be cpu, cuda or dml.")
        self.chat_template = chat_template if chat_template is not None else os.getenv("ONNX_CHAT_TEMPLATE")
        self.load_seconds = 0.0
        self.last_stats: GenerationStats | None = None
        self._runtime: Any = None
        self._model: Any = None
        self._native_tokenizer: Any = None
        self._processor: Any = None
        self._active: _Generation | None = None
        self._closed = False
        self._model_type = ""
        self._context_length = 0

    def _validate_assets(self) -> None:
        if self.model_path is None:
            raise RuntimeError(
                f"Set {self._path_env} in the root .env ({_ROOT_ENV}) or your environment "
                "to a local ONNX model folder containing genai_config.json; no model is downloaded."
            )
        if not self.model_path.is_dir():
            raise RuntimeError(
                f"{self._path_env} points to a missing model folder: {self.model_path}. "
                "Set it in the root .env to a complete local ONNX export folder, not an .onnx file."
            )
        config_path = self.model_path / "genai_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
                raise ValueError("model must be a configuration object")
            model_config = config["model"]
            self._model_type = model_config["type"]
            self._context_length = model_config["context_length"]
            if not isinstance(self._model_type, str):
                raise ValueError("model.type must be a string")
            if type(self._context_length) is not int or self._context_length <= 0:
                raise ValueError("model.context_length must be a positive integer")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Invalid or missing ONNX configuration at {config_path} ({self._path_env}): {exc}. "
                "Select a complete compatible model export folder in the root .env."
            ) from exc

        required = {"tokenizer.json"}

        def collect_assets(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"filename", "config_filename", "adapter_filename"} and item:
                        if not isinstance(item, str):
                            raise RuntimeError(f"ONNX asset {key} must be a filename string.")
                        required.add(item)
                    else:
                        collect_assets(item)
            elif isinstance(value, list):
                for item in value:
                    collect_assets(item)

        collect_assets(model_config)
        if self.vision:
            vision_config = model_config.get("vision")
            if self._model_type not in _VISION_TYPES or not isinstance(vision_config, dict):
                raise RuntimeError(
                    "Image input requires a compatible Phi-3/3.5 vision or Phi-4 multimodal ONNX model "
                    "with vision assets. A text-only phi-4-mini-instruct model cannot process images. "
                    "Set ONNX_MULTIMODAL_MODEL_PATH to the vision export."
                )
            if not vision_config.get("filename"):
                raise RuntimeError("The vision ONNX export is missing model.vision.filename.")
            required.add(vision_config.get("config_filename") or "processor_config.json")
        missing = sorted(name for name in required if not (self.model_path / name).is_file())
        if missing:
            raise RuntimeError(f"Incomplete ONNX export at {self.model_path}; missing assets: {', '.join(missing)}")

    def load(self) -> OnnxChatClient:
        """Lazily import the optional runtime and load the configured local model."""
        if self._closed:
            raise RuntimeError("This ONNX client is closed.")
        if self._model is not None:
            return self
        started = perf_counter()
        self._validate_assets()
        try:
            self._runtime = importlib.import_module("onnxruntime_genai")
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"ONNX Runtime GenAI is unavailable. {_INSTALL_GUIDANCE}") from exc
        version = getattr(self._runtime, "__version__", "0.0.0")
        try:
            parts = tuple(int(part) for part in version.split(".")[:3])
        except ValueError:
            parts = ()
        if parts < (0, 12, 0):
            raise RuntimeError(f"ONNX Runtime GenAI {MIN_GENAI_VERSION} or newer is required. {_INSTALL_GUIDANCE}")
        try:
            if self.provider != "cpu":
                available = getattr(self._runtime, f"is_{self.provider}_available")()
                if not available:
                    raise RuntimeError(f"The installed runtime does not support {self.provider}.")
            config = self._runtime.Config(str(self.model_path))
            config.clear_providers()
            if self.provider != "cpu":
                config.append_provider(self.provider)
            if self.provider == "cuda":
                config.set_provider_option("cuda", "device_id", "0")
                config.set_provider_option("cuda", "enable_cuda_graph", "0")
            self._model = self._runtime.Model(config)
            self._native_tokenizer = self._runtime.Tokenizer(self._model)
            if self.vision:
                self._processor = self._model.create_multimodal_processor()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Could not load ONNX model {self.model_path} with {self.provider}; "
                f"check compatible model/processor assets. {exc} {_INSTALL_GUIDANCE}"
            ) from exc
        self.load_seconds = perf_counter() - started
        return self

    def close(self) -> None:
        """Release model/processor/tokenizer and any partially consumed generation."""
        if self._active is not None:
            self._active.close()
            self._active = None
        self._processor = None
        self._native_tokenizer = None
        self._model = None
        self._runtime = None
        self._closed = True

    def __enter__(self) -> OnnxChatClient:
        return self.load()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _messages_and_images(self, messages: Sequence[Message]) -> tuple[list[dict[str, str]], list[bytes]]:
        if not messages:
            raise ValueError("At least one chat message is required.")
        turns: list[dict[str, str]] = []
        images: list[bytes] = []
        for message in messages:
            if message.role not in {"system", "user", "assistant"}:
                raise ValueError("ONNX supports only system, user and assistant messages, not tool calls.")
            pieces: list[str] = []
            markers: list[str] = []
            for content in message.contents:
                if content.type == "text":
                    pieces.append(content.text or "")
                elif content.type == "data" and (content.media_type or "").startswith("image/"):
                    if not self.vision:
                        raise RuntimeError("Image input requires vision=True and ONNX_MULTIMODAL_MODEL_PATH; it is not ignored.")
                    if message.role != "user":
                        raise ValueError("Phi vision images must be in user messages.")
                    prefix = f"data:{content.media_type};base64,"
                    if not content.uri or not content.uri.startswith(prefix):
                        raise ValueError("Image content must contain a base64 data URI.")
                    try:
                        image = base64.b64decode(content.uri[len(prefix):], validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise ValueError("Invalid base64 image data.") from exc
                    if not image:
                        raise ValueError("Image data is empty.")
                    images.append(image)
                    markers.append(f"<|image_{len(images)}|>\n")
                else:
                    raise ValueError("ONNX supports text and inline image data only; audio, remote URLs and tools are not ignored.")
            turns.append({"role": message.role, "content": "".join([*markers, *pieces])})
        return turns, images

    def _format_prompt(self, turns: list[dict[str, str]], image_count: int) -> str:
        if self.chat_template is not None:
            if len(turns) != 1 or turns[0]["role"] != "user" or "{prompt}" not in self.chat_template:
                raise ValueError("ONNX_CHAT_TEMPLATE must contain {prompt} and accepts a single user message only.")
            try:
                prompt = self.chat_template.format(prompt=turns[0]["content"])
            except (KeyError, ValueError) as exc:
                raise ValueError("Invalid ONNX_CHAT_TEMPLATE; use {prompt} and escape literal braces.") from exc
        else:
            template_path = self.model_path / "chat_template.jinja"
            template = template_path.read_text(encoding="utf-8") if template_path.is_file() else ""
            apply_template = getattr(self._native_tokenizer, "apply_chat_template", None)
            try:
                if apply_template is None:
                    raise RuntimeError("Tokenizer.apply_chat_template is unavailable")
                prompt = apply_template(
                    messages=json.dumps(turns), template_str=template, add_generation_prompt=True
                )
                if not isinstance(prompt, str) or not prompt:
                    raise RuntimeError("The native chat template returned an empty prompt")
            except Exception as exc:
                if template or self._model_type not in _PHI_TYPES:
                    raise RuntimeError(
                        "The model chat template could not be applied; provide a compatible "
                        "chat_template.jinja or single-turn ONNX_CHAT_TEMPLATE containing {prompt}."
                    ) from exc
                prompt = "".join(f"<|{turn['role']}|>\n{turn['content']}<|end|>\n" for turn in turns)
                prompt += "<|assistant|>\n"
        for index in range(1, image_count + 1):
            if f"<|image_{index}|>" not in prompt:
                raise RuntimeError("The chat template dropped a Phi vision image marker; image input cannot be ignored.")
        return prompt

    def _prepare_generation(
        self, state: _Generation, messages: Sequence[Message], options: Mapping[str, Any], stats: GenerationStats
    ) -> None:
        if options.get("instructions"):
            messages = [Message("system", [options["instructions"]]), *messages]
        turns, image_data = self._messages_and_images(messages)
        prompt = self._format_prompt(turns, len(image_data))
        if self.vision:
            try:
                if image_data:
                    state.images = self._runtime.Images.open_bytes(*image_data)
                state.inputs = self._processor(prompt, images=state.images)
                shape = state.inputs["input_ids"].shape
                shape = shape() if callable(shape) else shape
                if len(shape) != 2 or shape[0] != 1:
                    raise ValueError("Expected one processed input_ids sequence")
                stats.input_token_count = int(shape[1])
            except Exception as exc:
                raise RuntimeError(
                    "Could not process image input; check the image bytes and matching Phi vision "
                    f"ONNX/processor assets. {exc}"
                ) from exc
        else:
            input_tokens = self._native_tokenizer.encode(prompt)
            stats.input_token_count = len(input_tokens)
        if stats.input_token_count >= self._context_length:
            raise ValueError("The processed prompt fills the ONNX context; no room remains for output tokens.")
        requested = options.get("max_tokens", 512)
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
            raise ValueError("max_tokens must be a positive integer output-token budget.")
        stats.max_output_tokens = min(requested, self._context_length - stats.input_token_count)
        state.params = self._runtime.GeneratorParams(self._model)
        search: dict[str, Any] = {
            "max_length": stats.input_token_count + stats.max_output_tokens,
            "batch_size": 1,
            "num_beams": 1,
            "do_sample": False,
        }
        if options.get("temperature", 0) > 0:
            search.update(do_sample=True, temperature=options["temperature"])
        if options.get("top_p") is not None:
            search["top_p"] = options["top_p"]
            if options.get("temperature") != 0:
                search["do_sample"] = True
        if options.get("seed") is not None:
            search["random_seed"] = options["seed"]
        state.params.set_search_options(**search)
        state.decoder = self._native_tokenizer.create_stream()
        state.generator = self._runtime.Generator(self._model, state.params)
        if self.vision:
            state.generator.set_inputs(state.inputs)
        else:
            state.generator.append_tokens(input_tokens)

    async def _updates(
        self, messages: Sequence[Message], options: Mapping[str, Any], kwargs: Mapping[str, Any]
    ) -> AsyncIterator[ChatResponseUpdate]:
        options = await self._validate_options(options)
        unsupported = [key for key, value in options.items() if value is not None and key not in {
            "max_tokens", "instructions", "temperature", "top_p", "seed"
        }]
        if unsupported or kwargs:
            raise ValueError(f"Unsupported ONNX options: {', '.join([*unsupported, *kwargs])}")
        if self._active is not None:
            raise RuntimeError("An ONNX client supports one active generation at a time.")
        self.load()
        stats = GenerationStats(started_at=perf_counter())
        self.last_stats = stats
        state = _Generation()
        self._active = state
        response_id = uuid4().hex
        try:
            self._prepare_generation(state, messages, options, stats)
            while stats.output_token_count < stats.max_output_tokens:
                await asyncio.sleep(0)
                if self._closed:
                    raise RuntimeError("This ONNX client is closed.")
                if state.generator.is_done():
                    break
                state.generator.generate_next_token()
                tokens = state.generator.get_next_tokens()
                if len(tokens) != 1:
                    raise RuntimeError("Expected exactly one native output token for this single-sequence client.")
                now = perf_counter()
                stats.first_token_at = stats.first_token_at if stats.first_token_at is not None else now
                stats.last_token_at = now
                stats.output_token_count += 1
                yield ChatResponseUpdate(
                    contents=[Content.from_text(state.decoder.decode(int(tokens[0])))],
                    role="assistant", response_id=response_id, message_id=response_id,
                    model=str(self.model_path),
                )
            yield ChatResponseUpdate(
                contents=[Content.from_usage({
                    "input_token_count": stats.input_token_count,
                    "output_token_count": stats.output_token_count,
                    "total_token_count": stats.input_token_count + stats.output_token_count,
                })],
                role="assistant", response_id=response_id, message_id=response_id,
                model=str(self.model_path),
                finish_reason="length" if stats.output_token_count == stats.max_output_tokens else "stop",
            )
        finally:
            state.close()
            if self._active is state:
                self._active = None

    def _inner_get_response(
        self, *, messages: Sequence[Message], stream: bool, options: Mapping[str, Any], **kwargs: Any
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        updates = self._updates(messages, options, kwargs)
        response = ResponseStream(
            updates, finalizer=self._finalize_response_updates, cleanup_hooks=[updates.aclose]
        )
        if stream:
            return response
        return response.get_final_response()


async def stream_with_timings(
    client: OnnxChatClient, messages: Sequence[Message], *, max_output_tokens: int = 512
) -> GenerationStats:
    """Print streamed text and load/first-token/last-token/native-token throughput."""
    started = perf_counter()
    client.load()
    print("\nAssistant (ONNX ChatClient):")
    print(f"Load time: {client.load_seconds * 1000:.0f} ms")
    response = client.get_response(messages, stream=True, options={"max_tokens": max_output_tokens})
    first = True
    async for update in response:
        stats = client.last_stats
        if first and stats.first_token_at is not None:
            print(f"First token time: {(stats.first_token_at - started) * 1000:.0f} ms")
            first = False
        print(update.text, end="", flush=True)
    stats = client.last_stats
    print()
    if stats.last_token_at is None:
        print("First/last token time: n/a (no output tokens)")
    else:
        print(f"Last token time: {(stats.last_token_at - started) * 1000:.0f} ms")
    print(
        f"Speed: {stats.tokens_per_second:.0f} tokens/second "
        f"({stats.output_token_count} tokens in {stats.generation_seconds * 1000:.0f} ms)"
    )
    return stats
