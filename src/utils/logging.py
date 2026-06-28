"""
src/utils/logging.py
---------------------
Single get_logger() used everywhere instead of bare print() calls, so training
runs leave a record in results/logs/ as well as the console.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str = "two_tower", log_dir: str = None) -> logging.Logger:
    """
    Returns a logger that writes to stdout, and additionally to a timestamped
    file under `log_dir` if one is provided.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # avoid duplicate handlers if get_logger() is called more than once
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(log_dir, f"run_{timestamp}.log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {file_path}")

    return logger