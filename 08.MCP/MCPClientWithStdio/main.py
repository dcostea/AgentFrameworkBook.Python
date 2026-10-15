"""Discover MCP content, demonstrate progress/polling, then optionally run the robot agent."""

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import MCPStdioTool
from mcp import ClientSession, types
from mcp.client.stdio import stdio_client
from shared.mcp_examples import demonstrate, run_robot_agent, should_run_agent, stdio_parameters


def create_mcp_tool(session):
    params = stdio_parameters(__file__)
    return MCPStdioTool(name="Motors Client", command=params.command, args=params.args,
                        env=params.env, session=session, load_prompts=False, request_timeout=30)


async def main(provider="openai", *, read_input=input):
    async with stdio_client(stdio_parameters(__file__)) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30),
                                 client_info=types.Implementation(name="Motors Client", version="1.0.0")) as session:
            async with asyncio.timeout(10):
                await session.initialize()
            prompt = await demonstrate(session, stdio=True)
            if not should_run_agent(read_input):
                print("MCP demos complete; skipping the LLM agent.")
                return
            return await run_robot_agent(create_mcp_tool(session), prompt, provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    asyncio.run(main(parser.parse_args().provider))
