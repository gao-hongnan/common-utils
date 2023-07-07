"""Logger class for logging to console and file."""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

console = Console(
    theme=Theme(
        {
            "logging.level.debug": "magenta",
            "logging.level.info": "green",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
            "logging.level.critical": "bold red",
            # "logging.level.remark": "bold blue",
        }
    )
)


class CustomFormatter(logging.Formatter):
    """This class overrides logging.Formatter's pathname to be relative path."""

    def format(self, record):
        record.pathname = os.path.relpath(record.pathname)
        return super().format(record)


@dataclass
class Logger:
    """
    Class for logger. Consider using singleton design to maintain only one
    instance of logger (i.e., shared logger).

    This logger implementation adheres to several principles of good
    software design:

    1. Single Responsibility Principle (SRP): The logger's responsibility
       is clear - managing and configuring the logging process. It creates
       logging handlers, formatters, and manages logging files and paths.

    2. Open-Closed Principle (OCP): The logger is open for extension (e.g.,
       one can easily extend it to add different types of handlers) but
       closed for modification (adding new functionality doesn't require
       modification of the existing code).

    3. Liskov Substitution Principle (LSP): This principle is not explicitly
       implemented in this class since it doesn't have any subclasses.

    4. Interface Segregation Principle (ISP): The Logger class does not
       depend on any interfaces it doesn't use. The class itself can be
       seen as a high-level interface for logging operations.

    5. Dependency Inversion Principle (DIP): The logger depends on
       abstractions (e.g., the logging module's Handler and Formatter
       classes), not concrete classes.

    1. Factory Pattern: In terms of design patterns, this Logger class follows the 'Factory'
       pattern by creating and returning logging handlers and formatters.
       For more justification, just realize that factory pattern is a creational pattern
       which provides an interface for creating objects in a superclass, but
       allows subclasses to alter the type of objects that will be created.

       For more info, see my design pattern notes.

    Areas for improvement:

    1. Consider implementing a Singleton pattern to ensure only one
       instance of Logger is used throughout the application.

    2. Consider adding thread safety to ensure that logs from different
       threads don't interfere with each other.

    3. The logger could be further extended to support other types of
       logging, such as sending logs to an HTTP endpoint.


    Example:
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
        stream_handler = RichHandler(
            rich_tracebacks=True,
            level=self.level,
            show_level=True,
            show_path=True,
            show_time=True,
            markup=True,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
            console=console,
        )
        # FIXME: If you set custom formatter, it will duplicate level and time.
        # stream_handler.setFormatter(
        #     CustomFormatter(
        #         "%(asctime)s [%(levelname)s] %(pathname)s %(funcName)s L%(lineno)d: %(message)s",
        #         "%Y-%m-%d %H:%M:%S",
        #     )
        # )
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
