"""Decorator Functions."""
import time
from typing import Any, Callable, TypeVar
import numpy as np

from prettytable import PrettyTable
from rich.pretty import pprint

#  callable that takes any number of arguments and returns any value.
F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Timer decorator."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Create a table to display the results
        table = PrettyTable()
        table.field_names = ["Function Name", "Seconds", "Minutes", "Hours"]
        table.add_row(
            [
                func.__name__,
                f"{elapsed_time:.4f}",
                f"{elapsed_time / 60:.4f}",
                f"{elapsed_time / 60 / 60:.4f}",
            ]
        )

        pprint(table)
        return result

    return wrapper


@timer
def add_two_arrays(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Add two arrays together."""
    return array_1 + array_2


if __name__ == "__main__":
    array_1 = np.random.randint(0, 100, size=(10000, 10000))
    array_2 = np.random.randint(0, 100, size=(10000, 10000))
    add_two_arrays(array_1, array_2)
