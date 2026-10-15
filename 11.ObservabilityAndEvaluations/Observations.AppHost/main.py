"""Explicit-start Python launcher for a separately configured OTLP dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.observability_examples import apphost_main

if __name__ == "__main__":
    raise SystemExit(apphost_main())
