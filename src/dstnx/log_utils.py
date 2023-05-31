import logging
import sys
from datetime import datetime
from pathlib import Path

import dstnx


def get_time() -> str:
    return datetime.now().strftime("%Y_%m_%d")


class LoggingControl:
    def __init__(self, fp: Path):
        self.fp = fp
        self.time = get_time()
        self._init_handlers()
        self._init_formatters()

    def _init_handlers(self):
        logfile = self.fp / f"{self.time}.log"
        self.c_handler = logging.StreamHandler(stream=sys.stdout)
        self.f_handler = logging.FileHandler(logfile, mode="a")
        self.c_handler.setLevel(logging.INFO)
        self.f_handler.setLevel(logging.DEBUG)

    def _init_formatters(self):
        self.c_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.c_handler.setFormatter(self.c_format)
        self.f_handler.setFormatter(self.f_format)

    def add_handlers(self, logger: logging.Logger):
        # Add handlers to the logger
        logger.addHandler(self.c_handler)
        logger.addHandler(self.f_handler)


LOGGING_CONTROLLER = LoggingControl(fp=dstnx.fp.LOG_DST)


def get_logger(name: str):
    logger = logging.getLogger(name)
    LOGGING_CONTROLLER.add_handlers(logger)
    logger.setLevel(logging.DEBUG)
    return logger
