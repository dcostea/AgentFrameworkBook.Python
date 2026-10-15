"""Send local image or MP3 audio content to an Azure multimodal deployment."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Content, Message

from shared.chat_clients import ROBOT_SYSTEM, create_completion_client, create_sdk_client

DATA_DIR = Path(__file__).resolve().parent / "Data"


def build_messages(mode: str = "audio") -> list[Message]:
    if mode == "audio":
        # The Python Chat Completions adapter recognizes audio/mp3, not audio/mpeg.
        contents = [Content.from_data((DATA_DIR / "Task.mp3").read_bytes(), media_type="audio/mp3")]
    elif mode == "image":
        contents = [
            Content.from_text("Look at the image and proceed safely."),
            Content.from_data((DATA_DIR / "path.jpg").read_bytes(), media_type="image/jpeg"),
        ]
    else:
        raise ValueError("mode must be audio or image")
    return [Message("system", [ROBOT_SYSTEM]), Message("user", contents)]


async def main(mode: str = "audio") -> None:
    async with create_sdk_client("azure", audio=mode == "audio") as sdk:
        client = create_completion_client(sdk, "azure", audio=mode == "audio")
        response = await client.get_response(build_messages(mode))
        print(f"\nAssistant (Azure ChatClient): {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audio", "image"], default="audio")
    asyncio.run(main(parser.parse_args().mode))
