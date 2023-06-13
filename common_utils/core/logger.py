"""Logger class for logging to console and file."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

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

    log_output_dir: Optional[Union[str, Path]] = field(default=None, init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = self._init_logger()

    def _create_log_output_dir(self) -> Path:
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        log_output_dir = Path(self.log_dir) / current_time

        Path(log_output_dir).mkdir(parents=True, exist_ok=True)

        return log_output_dir

    def _get_log_file_path(self) -> Optional[Path]:
        if self.log_dir is not None:
            self.log_output_dir = self._create_log_output_dir()
            return self.log_output_dir / self.log_file
        return None

    def _create_stream_handler(self) -> RichHandler:
        stream_handler = RichHandler(rich_tracebacks=True, level=self.level)
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        return stream_handler

    def _create_file_handler(self, log_file_path: Path) -> logging.FileHandler:
        file_handler = logging.FileHandler(filename=str(log_file_path))
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        return file_handler

    def _init_logger(self) -> logging.Logger:
        # get module name, useful for multi-module logging
        logger = logging.getLogger(self.module_name or __name__)
        logger.setLevel(self.level)

        logger.addHandler(self._create_stream_handler())

        log_file_path = self._get_log_file_path()
        if log_file_path:
            logger.addHandler(self._create_file_handler(log_file_path))
        logger.propagate = self.propagate
        return logger
