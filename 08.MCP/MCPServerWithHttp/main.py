"""Serve the stateless Streamable HTTP motors MCP server on loopback."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.mcp_examples import MotorsServer, configured_port, configure_server_logging, port_number


def build_server(**kwargs):
    return MotorsServer(tasks_enabled=False, **kwargs)


def main(port=None):
    configure_server_logging()
    server = build_server(port=configured_port() if port is None else port_number(port))
    server.run(transport="streamable-http")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=configured_port())
    main(parser.parse_args().port)
