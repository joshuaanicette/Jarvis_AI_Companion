from __future__ import annotations

import argparse
import sys

from src.core.runtime import (
    JayRuntime,
    RuntimeMode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description=(
            "Jarvis local AI companion with "
            "terminal and browser interfaces"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            mode.value
            for mode in RuntimeMode
        ],
        default=RuntimeMode.HYBRID.value,
        help=(
            "Runtime mode: voice, text, hybrid, "
            "or once. Default: hybrid."
        ),
    )

    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help=(
            "Command to process when using "
            "--mode once."
        ),
    )

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help=(
            "Run Jarvis without opening the "
            "browser chat dashboard."
        ),
    )

    return parser


def start_chat_dashboard(
    runtime: JayRuntime,
    open_browser: bool = True,
) -> None:
    """
    Start the browser dashboard using the same
    Application instance owned by JayRuntime.
    """

    application = getattr(
        runtime,
        "app",
        None,
    )

    if application is None:
        raise RuntimeError(
            (
                "JayRuntime did not create "
                "an Application instance."
            )
        )

    dashboard = getattr(
        application,
        "chat_dashboard",
        None,
    )

    if dashboard is None:
        raise RuntimeError(
            (
                "Application does not have "
                "chat_dashboard configured."
            )
        )

    running = bool(
        getattr(
            dashboard,
            "running",
            False,
        )
    )

    if not running:
        dashboard.start()

    print(
        (
            "\nJarvis browser dashboard: "
            f"{dashboard.url}"
        ),
        flush=True,
    )

    if open_browser:
        open_method = getattr(
            dashboard,
            "open",
            None,
        )

        if callable(open_method):
            result = open_method()

            if result:
                print(
                    result,
                    flush=True,
                )

        else:
            print(
                (
                    "The dashboard is running, but "
                    "automatic browser opening is "
                    "not available."
                ),
                flush=True,
            )

            print(
                (
                    "Open this address manually: "
                    f"{dashboard.url}"
                ),
                flush=True,
            )


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
            (
                "--command is required when "
                "using --mode once"
            )
        )

    runtime: JayRuntime | None = None

    try:
        runtime = JayRuntime(
            mode=mode
        )

        if (
            not arguments.no_dashboard
            and mode != RuntimeMode.ONCE
        ):
            start_chat_dashboard(
                runtime=runtime,
                open_browser=True,
            )

        return runtime.start(
            initial_command=(
                arguments.command
            )
        )

    except KeyboardInterrupt:
        print(
            "\nStopping Jarvis...",
            flush=True,
        )

        return 0

    except Exception as error:
        print(
            (
                "Jarvis failed to start: "
                f"{error}"
            ),
            file=sys.stderr,
            flush=True,
        )

        return 1

    finally:
        if runtime is not None:
            application = getattr(
                runtime,
                "app",
                None,
            )

            state = getattr(
                application,
                "state",
                None,
            )

            still_online = bool(
                getattr(
                    state,
                    "online",
                    False,
                )
            )

            if (
                application is not None
                and still_online
            ):
                shutdown = getattr(
                    application,
                    "shutdown",
                    None,
                )

                if callable(shutdown):
                    try:
                        shutdown()

                    except Exception as error:
                        print(
                            (
                                "Jarvis shutdown "
                                f"warning: {error}"
                            ),
                            file=sys.stderr,
                            flush=True,
                        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )