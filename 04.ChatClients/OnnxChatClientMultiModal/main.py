"""Describe Data\\path.jpg with a compatible local Phi vision ONNX export.

Set ONNX_MULTIMODAL_MODEL_PATH to Phi-3/3.5 vision or Phi-4 multimodal, including
vision ONNX components, tokenizer and processor assets. The original C# active
phi-4-mini-instruct text-only model is NOT suitable. ONNX_IMAGE_PATH overrides
the sample's JPEG, and ONNX_PROVIDER defaults to CPU (CUDA is opt-in).
The client loads the root .env before resolving ONNX_IMAGE_PATH.
Install the optional requirements-onnx.txt dependencies; no assets are fetched.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Content, Message
from shared.onnx_client import OnnxChatClient, stream_with_timings

PROMPT = "Look at the image and tell what you see."


async def main() -> None:
    client = OnnxChatClient(vision=True)
    try:
        image_path = Path(os.getenv("ONNX_IMAGE_PATH") or str(Path(__file__).resolve().parent / "Data" / "path.jpg"))
        if not image_path.is_file():
            raise RuntimeError(f"JPEG input not found: {image_path}. Copy Data\\path.jpg or set ONNX_IMAGE_PATH.")
        message = Message("user", [Content.from_text(PROMPT), Content.from_data(image_path.read_bytes(), "image/jpeg")])
        await stream_with_timings(client, [message], max_output_tokens=512)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ONNX configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
