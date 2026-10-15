"""Configuration helpers shared by every sample.

This is the Python equivalent of the repeated C# block:

	var configuration = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
	var model = configuration["OpenAI:ModelId"];
	var apiKey = configuration["OpenAI:ApiKey"];

It reads OPENAI_API_KEY and OPENAI_CHAT_MODEL_ID from the environment
(or a local .env file) and builds an OpenAI chat client from them.
"""

import os
from typing import Any

from dotenv import load_dotenv

# Load variables from a .env file in the Agents/ root (if present).
load_dotenv()

_DEFAULT_MODEL = "gpt-5.4-mini"


class _ChatClientCreateAgentAdapter:
	"""Adapter to provide create_agent() over newer chat clients exposing as_agent()."""

	def __init__(self, chat_client: Any) -> None:
		self._chat_client = chat_client

	def create_agent(
		self,
		*,
		name: str | None = None,
		description: str | None = None,
		instructions: str | None = None,
		tools: Any | None = None,
		temperature: float | None = None,
		top_p: float | None = None,
		max_tokens: int | None = None,
		response_format: Any | None = None,
		**kwargs: Any,
	) -> Any:
		default_options: dict[str, Any] = {}
		if temperature is not None:
			default_options["temperature"] = temperature
		if top_p is not None:
			default_options["top_p"] = top_p
		if max_tokens is not None:
			default_options["max_tokens"] = max_tokens
		if response_format is not None:
			default_options["response_format"] = response_format

		as_agent_kwargs = {
			"name": name,
			"description": description,
			"instructions": instructions,
			"tools": tools,
		}
		if default_options:
			as_agent_kwargs["default_options"] = default_options

		# Preserve passthrough for future-compatible arguments.
		as_agent_kwargs.update(kwargs)
		return self._chat_client.as_agent(**as_agent_kwargs)

	def __getattr__(self, name: str) -> Any:
		return getattr(self._chat_client, name)


def create_chat_client() -> Any:
	"""Create a chat client that can also produce agents when supported.

	This preserves the sample ergonomics used by the agent-based projects while
	centralizing the shared environment and model lookup.
	"""
	try:
		from agent_framework.openai import OpenAIChatClient as openai_chat_client_class
	except ModuleNotFoundError:
		try:
			from agent_framework_openai import OpenAIChatClient as openai_chat_client_class
		except ModuleNotFoundError as exc:
			raise RuntimeError(
				"OpenAIChatClient could not be imported. Install dependencies with "
				"`pip install -r requirements.txt` and ensure `agent-framework` and "
				"`agent-framework-openai` are installed."
			) from exc

	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError(
			"OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
		)

	model = os.environ.get("OPENAI_CHAT_MODEL_ID", _DEFAULT_MODEL)
	
	chat_client = openai_chat_client_class(model=model, api_key=api_key)

	if hasattr(chat_client, "create_agent"):
		return chat_client

	if hasattr(chat_client, "as_agent"):
		return _ChatClientCreateAgentAdapter(chat_client)

	raise RuntimeError(
		"OpenAIChatClient does not provide create_agent/as_agent in this package "
		"version. Upgrade Agent Framework dependencies."
	)

