"""Run chapter 11 source datasets offline, or explicitly opt into model judges."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.evaluation_examples import evaluation_main

if __name__ == "__main__":
    raise SystemExit(asyncio.run(evaluation_main()))
