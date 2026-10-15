"""Chapter 09: shared chat-client middleware demonstration."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_sdk_client
from shared.chat_client_middleware_examples import build_demo, run_demo


async def main(provider: str = "openai", stream: bool = False) -> None:
    async with create_sdk_client(provider) as sdk:
        agent = build_demo(sdk, "shared", provider=provider)
        await run_demo(agent, "shared", stream=stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "azure"], default="openai")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.provider, args.stream))
