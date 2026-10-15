"""Discover MCP tools/resources/prompts over HTTP and run the robot agent."""

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import MCPStreamableHTTPTool
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from shared.mcp_examples import configured_port, demonstrate, http_url, port_number, run_robot_agent


def create_mcp_tool(session, port):
    return MCPStreamableHTTPTool(name="Motors Client", url=http_url(port), session=session,
                                 load_prompts=False, request_timeout=30)


async def main(provider="openai", port=None):
    port = configured_port() if port is None else port_number(port)
    async with streamable_http_client(http_url(port)) as (read, write, _):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30),
                                 client_info=types.Implementation(name="Motors Client", version="1.0.0")) as session:
            async with asyncio.timeout(10):
                await session.initialize()
            prompt = await demonstrate(session)
            return await run_robot_agent(create_mcp_tool(session, port), prompt, provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    parser.add_argument("--port", type=port_number, default=configured_port())
    args = parser.parse_args()
    asyncio.run(main(args.provider, args.port))
