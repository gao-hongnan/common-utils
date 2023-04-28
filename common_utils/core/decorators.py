"""Decorator Functions."""
import time
from typing import Any, Callable, TypeVar
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
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    add(1, 2)
