"""
Logging configuration for the project.

Logs are written to:
1. Console
2. logs/app.log

Usage in modules:
    import logging
    from config.logging_config import setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure application-wide logging.

    This function should be called once at application entry points,
    such as classification/pipeline.py or backend/app/main.py.
    """

    LOG_DIR.mkdir(exist_ok=True)

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers if setup_logging() is called more than once
    if root_logger.handlers:
        return

    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
