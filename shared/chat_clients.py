"""Provider configuration and prompts shared by the book's chapter samples."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROBOT_SYSTEM = """## Persona
You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.

## Action
You have to break down the provided complex commands into the basic moves you know.

## Template
Respond with a JSON array like [move1, move2, move3].
Do not respond with reasoning, comments, or any additional text."""

TREE_COMMAND = "There is a tree in front of the car. Avoid it and resume the original path."
ROBOT_PROMPT = f"{ROBOT_SYSTEM}\n\n## Complex Command\n{TREE_COMMAND}"
BACKGROUND_PROMPT = """You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Respond with the moves and their parameters (angle or distance), and provide detailed additional explanations.

Complex command:
"There is a tree directly in front of the car. Avoid it and then return to the original path."
"""
TOOLS_SYSTEM = """You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into the basic moves you know.
Do not respond with reasoning, comments, or any additional text."""


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Configure it in the root .env file or environment.")
    return value


def model_for(provider: str = "openai", *, audio: bool = False, responses: bool = False) -> str:
    if provider == "azure":
        if audio:
            return require_env("AZURE_OPENAI_AUDIO_DEPLOYMENT_NAME")
        if responses and os.environ.get("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"):
            return require_env("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")
        return require_env("AZURE_OPENAI_DEPLOYMENT_NAME")
    if provider == "openai":
        if responses and os.environ.get("OPENAI_RESPONSES_MODEL_ID"):
            return require_env("OPENAI_RESPONSES_MODEL_ID")
        return os.environ.get("OPENAI_CHAT_MODEL_ID") or "gpt-5.4-mini"
    raise ValueError(f"Unknown provider: {provider}")


def azure_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return endpoint + "/"
    if endpoint.endswith("/openai"):
        return endpoint + "/v1/"
    return endpoint + "/openai/v1/"


def create_sdk_client(provider: str = "openai", *, audio: bool = False) -> AsyncOpenAI:
    """Use the Azure v1 endpoint by default; opt into legacy routing with an API version."""
    if provider == "openai":
        kwargs = {"api_key": require_env("OPENAI_API_KEY")}
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = require_env("OPENAI_BASE_URL")
        return AsyncOpenAI(**kwargs)
    if provider != "azure":
        raise ValueError(f"Unknown provider: {provider}")
    prefix = "AZURE_OPENAI_AUDIO" if audio else "AZURE_OPENAI"
    endpoint = require_env(f"{prefix}_ENDPOINT")
    api_key = require_env(f"{prefix}_API_KEY")
    api_version = os.environ.get(f"{prefix}_API_VERSION") or os.environ.get("AZURE_OPENAI_API_VERSION")
    if api_version:
        resource_endpoint = endpoint.rstrip("/").removesuffix("/openai/v1").removesuffix("/openai")
        return AsyncAzureOpenAI(api_key=api_key, azure_endpoint=resource_endpoint, api_version=api_version)
    return AsyncOpenAI(api_key=api_key, base_url=azure_base_url(endpoint))


def create_completion_client(sdk: AsyncOpenAI, provider: str = "openai", *, audio: bool = False, middleware=None):
    from agent_framework.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=model_for(provider, audio=audio), async_client=sdk, middleware=middleware,
    )


def create_responses_client(sdk: AsyncOpenAI, provider: str = "openai"):
    from agent_framework.openai import OpenAIChatClient

    return OpenAIChatClient(model=model_for(provider, responses=True), async_client=sdk)
