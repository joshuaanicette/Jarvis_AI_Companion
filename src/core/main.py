import argparse
import sys

from src.core.runtime import (
    JayRuntime,
    RuntimeMode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="joe",
        description="Joe local AI companion runtime",
    )

    parser.add_argument(
        "--mode",
        choices=[
            mode.value
            for mode in RuntimeMode
        ],
        default=RuntimeMode.VOICE.value,
        help=(
            "Runtime mode: voice, text, hybrid, or once. "
            "The default mode is voice."
        ),
    )

    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help=(
            "Command to process when using --mode once."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()

    mode = RuntimeMode(
        arguments.mode
    )

    if (
        mode == RuntimeMode.ONCE
        and not arguments.command
    ):
        parser.error(
            "--command is required with --mode once"
        )

    runtime = JayRuntime(
        mode=mode
    )

    return runtime.start(
        initial_command=arguments.command
    )


if __name__ == "__main__":
    sys.exit(main())