from __future__ import annotations

import sys
import time

from src.core.application import (
    Application,
)


def main() -> int:
    """
    Run the Jarvis browser dashboard as a standalone test.

    For the final terminal and browser combination, run main.py
    instead so both interfaces use one Application instance.
    """

    application: (
        Application
        | None
    ) = None

    try:
        application = Application()

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

        if not dashboard.running:
            dashboard.start()

        result = dashboard.open()

        print(
            result,
            flush=True,
        )

        print(
            (
                "Jarvis dashboard URL: "
                f"{dashboard.url}"
            ),
            flush=True,
        )

        print(
            (
                "Press Ctrl+C to stop "
                "the standalone dashboard."
            ),
            flush=True,
        )

        while True:
            time.sleep(
                1.0
            )

    except KeyboardInterrupt:
        print(
            (
                "\nStopping the Jarvis "
                "dashboard..."
            ),
            flush=True,
        )

        return 0

    except Exception as error:
        print(
            (
                "Jarvis dashboard failed: "
                f"{error}"
            ),
            file=sys.stderr,
            flush=True,
        )

        return 1

    finally:
        if application is not None:
            try:
                application.shutdown()

            except Exception as error:
                print(
                    (
                        "Jarvis shutdown error: "
                        f"{error}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )