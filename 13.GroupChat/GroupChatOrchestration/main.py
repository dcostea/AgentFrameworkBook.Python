"""MotorsAgent → NavigatorAgent: refine, approve, execute (at most 10 turns)."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.group_chat_examples import provider_from_cli, run_example


async def main(provider="openai"):
    return await run_example("GroupChatOrchestration", provider=provider)


if __name__ == "__main__":
    asyncio.run(main(provider_from_cli()))
