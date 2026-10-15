"""Send the original local map; optionally use the other image or MP3 assets."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent, Content, Message
from shared.agent_examples import JSON_INSTRUCTIONS
from shared.chat_clients import create_completion_client, create_sdk_client

DATA_DIR = Path(__file__).resolve().parent / "Data"
ASSET_MEDIA_TYPES = {
    "Map.png": "image/png",
    "path.jpg": "image/jpeg",
    "road_and_tree.png": "image/png",
    # In 1.8.1 the chat adapter recognizes MP3 input by this MIME alias.
    "Task.mp3": "audio/mp3",
}


def build_conversation(data_dir: Path = DATA_DIR, *, asset: str = "Map.png") -> list[Message]:
    if asset not in ASSET_MEDIA_TYPES:
        raise ValueError(f"Unknown asset: {asset}. Choose one of {', '.join(ASSET_MEDIA_TYPES)}")
    media_type = ASSET_MEDIA_TYPES[asset]
    prompt = "Follow the spoken task safely." if media_type.startswith("audio/") else (
        "Look at the image of the map and proceed safely."
    )
    return [Message("user", [
        Content.from_text(prompt),
        Content.from_data((data_dir / asset).read_bytes(), media_type),
    ])]


async def run_example(agent: Agent, *, data_dir: Path = DATA_DIR, asset: str = "Map.png"):
    response = await agent.run(build_conversation(data_dir, asset=asset))
    print(response.text)
    return response


async def main(provider: str = "openai", *, asset: str = "Map.png") -> None:
    audio = asset == "Task.mp3"
    async with create_sdk_client(provider, audio=audio) as sdk:
        agent = Agent(
            client=create_completion_client(sdk, provider, audio=audio), instructions=JSON_INSTRUCTIONS,
        )
        await run_example(agent, asset=asset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=ASSET_MEDIA_TYPES, default="Map.png")
    asyncio.run(main(asset=parser.parse_args().asset))
