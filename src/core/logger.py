import logging
from pathlib import Path


def setup_logger(name: str = "Jay") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    Path("logs").mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler("logs/jay.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()
