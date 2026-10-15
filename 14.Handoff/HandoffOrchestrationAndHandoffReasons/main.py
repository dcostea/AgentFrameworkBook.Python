"""Sensor-based robot handoff with edge-description routing rules."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.chat_clients import create_sdk_client
from shared.handoff_examples import SensorReadings, build_workflow, run_workflow


async def main(provider="openai", scenario="random"):
    async with create_sdk_client(provider) as sdk:
        workflow = build_workflow(sdk, provider=provider, reasons=True, readings=SensorReadings.scenario(scenario))
        return await run_workflow(workflow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "azure"], default="openai")
    parser.add_argument("--scenario", choices=["random", "safe", "fire", "rain"], default="random")
    args = parser.parse_args()
    result = asyncio.run(main(args.provider, args.scenario))
    if result.get_request_info_events() or not result.get_outputs():
        raise SystemExit(1)
