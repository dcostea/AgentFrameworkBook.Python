"""Structured Python logging for agent runs and simulated tool timings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.agent_middleware_examples import main_for


def main(argv=None):
    main_for("AgentWithLogging", argv)


if __name__ == "__main__":
    main()
