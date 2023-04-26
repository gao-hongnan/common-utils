"""Decorator Functions."""
import time
from typing import Any, Callable, TypeVar

#  callable that takes any number of arguments and returns any value.
F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Timer decorator."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        print(f"{func.__name__} took {elapsed_time / 60:.4f} minutes to execute.")
        print(f"{func.__name__} took {elapsed_time / 60 / 60:.4f} hours to execute.")
        return result

    return wrapper
