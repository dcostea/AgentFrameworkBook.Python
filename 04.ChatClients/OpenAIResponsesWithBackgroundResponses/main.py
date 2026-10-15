"""Create and poll an OpenAI background response using continuation tokens."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.background_responses import run_background_response
from shared.chat_clients import create_responses_client, create_sdk_client


async def main() -> None:
    async with create_sdk_client() as sdk:
        await run_background_response(create_responses_client(sdk))


if __name__ == "__main__":
    asyncio.run(main())
