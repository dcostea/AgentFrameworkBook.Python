"""Run the chapter 12 ConcurrentOrchestration source graph."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.sequential_concurrent_examples import main


if __name__ == "__main__":
    raise SystemExit(main("ConcurrentOrchestration", sample_dir=Path(__file__).resolve().parent))
