"""Bounded continuation-token polling for the Responses API samples."""

import asyncio
from typing import Any

from agent_framework import Message

from .chat_clients import BACKGROUND_PROMPT


async def run_background_response(client: Any, *, max_attempts: int = 60, delay: float = 1.0):
    if max_attempts < 1 or delay < 0:
        raise ValueError("max_attempts must be positive and delay nonnegative")
    print(f"User:\n{BACKGROUND_PROMPT}")
    response = await client.get_response(
        [Message("user", [BACKGROUND_PROMPT])], options={"background": True, "store": True},
    )
    print("\n[INITIAL] Assistant:")
    print(f"Finish Reason: {response.finish_reason or '(null)'}")
    print(f"Has Continuation Token: {response.continuation_token is not None}")
    print(response.text)
    if response.continuation_token is not None:
        print("POLLING", end="", flush=True)
        for _ in range(max_attempts):
            await asyncio.sleep(delay)
            print(".", end="", flush=True)
            response = await client.get_response(
                [], options={"continuation_token": response.continuation_token},
            )
            if response.continuation_token is None:
                break
        else:
            token = response.continuation_token
            response_id = token.get("response_id", "unknown") if isinstance(token, dict) else str(token)
            raise TimeoutError(
                f"Polling timeout after {max_attempts} attempts. Response {response_id} is still running; "
                "retrieve or cancel it with the provider's Responses API."
            )
    print("\n\n[FINAL] Assistant:")
    print(f"Finish Reason: {response.finish_reason or '(null)'}")
    print(response.text)
    raw = response.raw_representation
    status = getattr(raw, "status", None)
    if status in {"failed", "cancelled", "incomplete"}:
        detail = getattr(raw, "error", None) or getattr(raw, "incomplete_details", None)
        raise RuntimeError(f"Background response ended with status {status}: {detail}")
    return response
