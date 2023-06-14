"""Logger class for logging to console and file."""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from rich.logging import RichHandler


class CustomFormatter(logging.Formatter):
    """This class overrides logging.Formatter's pathname to be relative path."""

    def format(self, record):
        record.pathname = os.path.relpath(record.pathname)
        return super().format(record)


@dataclass
class Logger:
    """Class for logger. Consider using singleton design to
    maintain only one instance of logger (i.e. shared logger).

    Logger(
        log_file="pipeline_training.log",
        module_name=__name__,
        level=logging.INFO,
        propagate=False,
        log_root_dir="/home/user/logs",
    )
    --> produces the below tree structure, note the log_root_dir is the root
        while the session_log_dir is the directory for the current session.
    /home/
    │
    └───user/
        │
        └───logs/                        # This is the log_root_dir
            │
            └───2023-06-14T10:20:30/     # This is the session_log_dir
                │
                └───pipeline_training.log


    Parameters
    ----------
    log_file : Optional[str], default=None
        The name of the log file. Logs will be written to this file if specified.
        It must be specified if `log_root_dir` is specified.
    module_name : Optional[str], default=None
        The name of the module. This is useful for multi-module logging.
    level : int, default=logging.INFO
        The level of logging.
    propagate : bool, default=False
        Whether to propagate the log message to parent loggers.
    log_root_dir : Optional[str], default=None
        The root directory for all logs. If specified, a subdirectory will be
        created in this directory for each logging session, and the log file will
        be created in the subdirectory. Must be specified if `log_file` is specified.

    Attributes
    ----------
    session_log_dir : Optional[Union[str, Path]]
        The directory for the current logging session. This is a subdirectory
        within `log_root_dir` that is named with the timestamp of when the logger
        was created.
    logger : logging.Logger
        The logger instance.

    Raises
    ------
    AssertionError
        Both `log_file` and `log_root_dir` must be provided, or neither should be provided.
    """

    log_file: Optional[str] = None
    module_name: Optional[str] = None
    level: int = logging.INFO
    propagate: bool = False
    log_root_dir: Optional[str] = None

    session_log_dir: Optional[Union[str, Path]] = field(default=None, init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        assert (self.log_file is None and self.log_root_dir is None) or (
            self.log_file is not None and self.log_root_dir is not None
        ), "Both log_file and log_root_dir must be provided, or neither should be provided."

        self.logger = self._init_logger()

    def _create_log_output_dir(self) -> Path:
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        session_log_dir = Path(self.log_root_dir) / current_time

        Path(session_log_dir).mkdir(parents=True, exist_ok=True)

        return session_log_dir

    def _get_log_file_path(self) -> Optional[Path]:
        if self.log_root_dir is not None:
            self.session_log_dir = self._create_log_output_dir()
            return self.session_log_dir / self.log_file
        return None

    def _create_stream_handler(self) -> RichHandler:
        stream_handler = RichHandler(rich_tracebacks=True, level=self.level)
        stream_handler.setFormatter(
            CustomFormatter(
                "%(asctime)s [%(levelname)s] %(pathname)s %(funcName)s L%(lineno)d: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        return stream_handler

    def _create_file_handler(self, log_file_path: Path) -> logging.FileHandler:
        file_handler = logging.FileHandler(filename=str(log_file_path))
        file_handler.setFormatter(
            CustomFormatter(
                "%(asctime)s [%(levelname)s] %(pathname)s %(funcName)s L%(lineno)d: %(message)s",
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
