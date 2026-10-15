"""Chainable safety and audit middleware for motor tool invocations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.agent_middleware_examples import main_for


def main(argv=None):
    main_for("AgentFunctionCallingMiddleware", argv)


if __name__ == "__main__":
    main()
