"""Decorator Functions."""
import functools
import time
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, TypeVar

import numpy as np
from prettytable import PrettyTable
from rich.pretty import pprint

from common_utils.core.logger import Logger

# Setup logging
LOGGER = Logger(
    module_name=__name__, propagate=False, log_root_dir=None, log_file=None
).logger


F = TypeVar("F", bound=Callable[..., Any])


# probably the only time when you don't use CamelCase for class names
class timer:
    def __init__(
        self,
        display_table: bool = True,
        unit: str = "seconds",
        decimal_places: int = 4,
        store_times: bool = False,
        log: bool = False,
    ) -> None:
        self.display_table = display_table
        self.unit = unit
        self.decimal_places = decimal_places
        self.store_times = store_times
        self.execution_times = {}
        self.log = log

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time = convert_time_unit(elapsed_time, self.unit)

            if self.store_times:
                if func.__name__ not in self.execution_times:
                    self.execution_times[func.__name__] = []
                self.execution_times[func.__name__].append(elapsed_time)

            if self.display_table:
                # Create a table to display the results
                table = PrettyTable()
                table.field_names = [
                    "Function Name",
                    f"Elapsed Time ({self.unit.capitalize()})",
                ]
                table.add_row(
                    [
                        func.__name__,
                        f"{round(elapsed_time, self.decimal_places):.{self.decimal_places}f}",
                    ]
                )

                if self.store_times and len(self.execution_times[func.__name__]) > 1:
                    times = self.execution_times[func.__name__]
                    stats_table = PrettyTable()
                    stats_table.field_names = ["Mean", "Median", "Stdev"]
                    stats_table.add_row(
                        [
                            round(mean(times), self.decimal_places),
                            round(median(times), self.decimal_places),
                            round(stdev(times), self.decimal_places),
                        ]
                    )
                    table.add_row(["Statistics", stats_table.get_string()])

                pprint(table)

            # Log the results
            if self.log:
                LOGGER.info(
                    f"Function {func.__name__} executed in "
                    f"{elapsed_time} seconds with args: {args} and "
                    f"kwargs: {kwargs}"
                )
            return result

        return wrapper


def convert_time_unit(time_in_seconds: float, unit: str) -> float:
    """Converts time to the desired unit."""
    if unit == "minutes":
        return time_in_seconds / 60
    if unit == "hours":
        return time_in_seconds / 60 / 60
    if unit == "seconds":
        return time_in_seconds

    raise ValueError("Unknown time unit. Please use 'seconds', 'minutes', or 'hours'.")


@timer(
    display_table=True, unit="seconds", decimal_places=4, store_times=True, log=False
)
def add_two_arrays(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Add two arrays together."""
    return array_1 + array_2


if __name__ == "__main__":
    array_1 = np.random.randint(0, 100, size=(10000, 10000))
    array_2 = np.random.randint(0, 100, size=(10000, 10000))
    REPEAT = 1
    for _ in range(REPEAT):
        add_two_arrays(array_1, array_2)
