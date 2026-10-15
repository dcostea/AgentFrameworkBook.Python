"""Compose shared-function, completed-response and function middleware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.agent_middleware_examples import main_for


def main(argv=None):
    main_for("AgentMiddlewareMixed", argv)


if __name__ == "__main__":
    main()
