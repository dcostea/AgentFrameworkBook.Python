"""Serve simulated motors, resources, prompts and experimental MCP Tasks over stdio."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.mcp_examples import MotorsServer, configure_server_logging


def build_server(**kwargs):
    return MotorsServer(tasks_enabled=True, **kwargs)


def main():
    configure_server_logging()
    print("Starting MCP Server running with Stdio transport type", file=sys.stderr, flush=True)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
