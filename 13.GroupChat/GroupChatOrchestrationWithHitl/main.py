"""Motors → Navigator → Human: human-authoritative approval in the workflow graph."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.group_chat_examples import provider_from_cli, run_example


async def main(provider="openai", *, read_input=input):
    return await run_example("GroupChatOrchestrationWithHitl", provider=provider, read_input=read_input)


if __name__ == "__main__":
    asyncio.run(main(provider_from_cli()))
