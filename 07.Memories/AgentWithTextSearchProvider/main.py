"""Keyword RAG with exact source matching and six recent user query texts.

The four policy documents are active; the source's commented-out weather
entries are deliberately not included. Retrieval memory is not chat history.
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework import Agent, InMemoryHistoryProvider
from shared.chat_clients import create_completion_client, create_sdk_client
from shared.memory_providers import (
    CustomKeywordSearchAdapter, TextSearchProvider, SearchItem,
    ROBOT_INSTRUCTIONS, ROBOT_PROMPTS, print_history, run_prompts,
)

KNOWLEDGE = (
    SearchItem("1", "The robot car can move backward by a specified distance. Reverse motion is limited to 5 meters.",
               "RobotCar Movement Policy", ("command", "move", "distance")),
    SearchItem("2", "The robot can turn left or right by angles of 30, 45, and 60 degrees.",
               "RobotCar Turning Policy", ("command", "move", "turn", "angle")),
    SearchItem("3", "The robot car can move forward by a specified distance. Maximum range is 10 meters.",
               "RobotCar Movement Policy", ("command", "move", "distance")),
    SearchItem("4", "Emergency stop immediately halts all motion.",
               "RobotCar Safety Manual", ("command", "move", "distance", "emergency")),
)


def build_agent(client):
    return Agent(
        client=client, name="RobotCarAgent", instructions=ROBOT_INSTRUCTIONS,
        description="An agent that assists a robot with the basic moves.",
        context_providers=[
            TextSearchProvider(CustomKeywordSearchAdapter(KNOWLEDGE, top_results=5), recent_message_memory_limit=6),
            InMemoryHistoryProvider("conversation"),
        ],
    )


async def main(provider="openai"):
    async with create_sdk_client(provider) as sdk:
        agent = build_agent(create_completion_client(sdk, provider))
        session = agent.create_session()
        await run_prompts(agent, session, ROBOT_PROMPTS)
        print_history(session.state["conversation"]["messages"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), default="openai")
    asyncio.run(main(parser.parse_args().provider))
