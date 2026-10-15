"""Run the local OpenTelemetry robot simulation, or opt into a live agent."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.observability_examples import observation_main

if __name__ == "__main__":
    raise SystemExit(asyncio.run(observation_main()))
