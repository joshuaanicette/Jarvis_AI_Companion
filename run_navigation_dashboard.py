import time

from src.location.location_manager import LocationManager
from src.navigation.dashboard import NavigationDashboard


def main() -> None:
    manager = LocationManager()

    dashboard = NavigationDashboard(
        location_manager=manager,
        host="127.0.0.1",
        port=8770,
    )

    dashboard.start()
    dashboard.open()

    print(
        f"Jay Navigation is running at {dashboard.url}"
    )

    try:
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        dashboard.stop()


if __name__ == "__main__":
    main()
