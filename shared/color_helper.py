"""Console color helpers - Python mirror of the C# Helpers/ColorHelper.cs.

Uses ANSI escape codes so the output is colored in any modern terminal,
mirroring the ConsoleColor-based PrintColored / PrintColoredLine methods.
"""

from enum import Enum


class Color(str, Enum):
	"""A small subset of console colors used by the samples."""

	BLACK = "30"
	RED = "31"
	GREEN = "32"
	YELLOW = "33"
	BLUE = "34"
	MAGENTA = "35"
	CYAN = "36"
	WHITE = "37"


_RESET = "\033[0m"


def print_colored_line(text: str, color: Color = Color.WHITE) -> None:
	"""Write text in the given color followed by a newline, then reset."""
	print(f"\033[{color.value}m{text}{_RESET}")


def print_colored(text: str, color: Color = Color.WHITE) -> None:
	"""Write text in the given color without a trailing newline, then reset."""
	print(f"\033[{color.value}m{text}{_RESET}", end="")
