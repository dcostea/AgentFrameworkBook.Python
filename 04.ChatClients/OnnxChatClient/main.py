"""Stream robot commands from a local ONNX model (no downloads or server).

Set ONNX_MODEL_PATH to a complete GenAI export and optionally ONNX_PROVIDER
(cpu by default; cuda selects device 0, enable_cuda_graph=0). Models without
an embedded chat template need chat_template.jinja or ONNX_CHAT_TEMPLATE;
e.g. Mistral v0.2 single-turn: [INST] {prompt} [/INST]. See shared.onnx_client.
The client explicitly loads the root .env; existing environment values win.
Optional runtime dependencies are in requirements-onnx.txt.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Message
from shared.onnx_client import OnnxChatClient, stream_with_timings

PROMPT = '''You are an AI assistant controlling a robot car capable of performing basic moves: forward, backward, turn left, turn right, and stop.
You have to break down the provided complex commands into basic moves you know.
Respond only with the permitted moves, without any additional explanations.

Complex command:
"There is a tree directly in front of the car. Avoid it and then come back to the original path."'''


async def main() -> None:
    print(f"USER: {PROMPT}")
    client = OnnxChatClient()
    try:
        await stream_with_timings(client, [Message("user", [PROMPT])], max_output_tokens=512)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ONNX configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
