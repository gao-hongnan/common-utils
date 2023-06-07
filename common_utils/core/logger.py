"""Logger class for logging to console and file."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


@dataclass
class Logger:
    """Class for logger. Consider using singleton design to
    maintain only one instance of logger (i.e. shared logger)."""

    log_file: Optional[str] = "log.txt"
    module_name: Optional[str] = None
    level: int = logging.INFO
    propagate: bool = False
    log_dir: Optional[str] = "outputs"

    log_output_dir: Optional[str] = field(default=None, init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = self._init_logger()

    def _create_log_output_dir(self) -> Path:
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        log_output_dir = Path(self.log_dir) / current_time

        Path(log_output_dir).mkdir(parents=True, exist_ok=True)

        return log_output_dir

    def _init_logger(self) -> logging.Logger:
        log_output_dir = self._create_log_output_dir()
        log_file_path = log_output_dir / self.log_file if self.log_file else None

        if self.module_name is None:
            logger = logging.getLogger(__name__)
        else:
            # get module name, useful for multi-module logging
            logger = logging.getLogger(self.module_name)

        logger.setLevel(self.level)

        stream_handler = RichHandler(rich_tracebacks=True)
        logger.addHandler(stream_handler)

        if log_file_path:
            file_handler = logging.FileHandler(filename=str(log_file_path))
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s]: %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )

            logger.addHandler(file_handler)

        logger.propagate = self.propagate

        self.log_output_dir = log_output_dir
        return logger
