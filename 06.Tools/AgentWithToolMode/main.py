"""Compare automatic, disabled, required, and specific motor-tool selection."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_sdk_client
from shared.tool_examples import build_agent


def tool_choice(mode: str):
    if mode in {"auto", "none", "required"}:
        return mode
    if mode == "backward":
        return {"mode": "required", "required_function_name": "backward"}
    raise ValueError(f"Unknown tool mode: {mode}")


async def main(mode: str = "auto") -> None:
    async with create_sdk_client() as sdk:
        agent = build_agent(sdk, options={"tool_choice": tool_choice(mode)})
        for prompt in ("Go left and right then stop.", "What movements can you perform?"):
            print(f"USER: {prompt}")
            response = await agent.run(prompt)
            print(f"RESPONSE:\n{response.text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "none", "required", "backward"], default="auto")
    asyncio.run(main(parser.parse_args().mode))
