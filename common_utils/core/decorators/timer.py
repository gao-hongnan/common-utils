"""Decorator Functions."""
import functools
import time
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from rich.pretty import pprint

from common_utils.core.common import seed_all

seed_all(1992)

DataTypes = Union[List[int], Dict[int, int], None]
#  callable that takes any number of arguments and returns any value.
F = TypeVar("F", bound=Callable[..., Any])


# pylint: disable=invalid-name
def data_factory(data_type: str, n: int) -> DataTypes:
    if data_type == "array":
        return list(range(n))
    if data_type == "dict":
        return {i: i for i in range(n)}
    if data_type is None:
        return None
    raise ValueError(f"Invalid data_type: {data_type}")


def time_complexity(
    data_type: str, repeat: int = 1, plot: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Tuple]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Tuple]:
        def wrapper(n_sizes: List[int], *args: Any, **kwargs: Dict[str, Any]) -> Tuple:
            avg_times = []
            median_times = []
            best_times = []
            worst_times = []

            for n in n_sizes:
                # create a list of n elements
                data_structure = data_factory(data_type, n)
                # note array is created outside the loop
                runtimes = []
                for _ in range(repeat):
                    start_time = time.perf_counter()

                    # pylint: disable=expression-not-assigned,line-too-long
                    func(n, data_structure, *args, **kwargs) if data_type else func(
                        n, *args, **kwargs
                    )  # <--- this is where it calls the function with n or data_structure as argument
                    end_time = time.perf_counter()
                    runtimes.append(end_time - start_time)

                avg_times.append(np.mean(runtimes))
                median_times.append(np.median(runtimes))
                best_times.append(np.min(runtimes))
                worst_times.append(np.max(runtimes))

            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(n_sizes, avg_times, "o-", label="Average")
                plt.plot(n_sizes, median_times, "o-", label="Median")
                plt.plot(n_sizes, best_times, "o-", label="Best")
                plt.plot(n_sizes, worst_times, "o-", label="Worst")
                plt.xlabel("Size of Input (n)")
                plt.ylabel("Execution Time (s)")
                plt.legend()
                plt.grid(True)
                plt.title(f"Time Complexity of {func.__name__}")
                plt.show()

            return n_sizes, avg_times, median_times, best_times, worst_times

        return wrapper

    return decorator


def timer(func: F) -> F:
    """Timer decorator."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Any:
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


@time_complexity(data_type="array", repeat=10, plot=True)
def list_access(n: int, array) -> None:
    _ = array[n // 2]


@time_complexity(data_type="array", repeat=10, plot=True)
def list_append(n: int, array) -> None:
    array.append(n)


@time_complexity(data_type="array", repeat=10, plot=True)
def list_insert(n: int, array) -> None:
    array.insert(0, n)


@time_complexity(data_type="array", repeat=10, plot=True)
def list_search(n: int, array) -> None:
    _ = n in array


# @time_complexity(repeat=10, plot=True)
# def for_loop(n: int, array) -> None:
#     for i in range(n):


@time_complexity(data_type="dict", repeat=10, plot=True)
def dict_set(n: int, dict_) -> None:
    dict_[n] = n


@time_complexity(data_type="dict", repeat=10, plot=True)
def dict_search(n: int, dict_) -> None:
    _ = n in dict_


@time_complexity(data_type=None, repeat=10, plot=True)
def for_loop(n: int) -> None:
    for _ in range(n):
        pass


@timer
def add_two_arrays(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Add two arrays together."""
    return array_1 + array_2


# list_access(range(1000000, 10000001, 1000000))
# list_append(range(1000000, 10000001, 1000000))
# list_insert(range(1000000, 10000001, 1000000))
# list_search(range(1000000, 10000001, 1000000))

dict_set(range(1000000, 10000001, 1000000))
dict_search(range(1000000, 10000001, 1000000))
for_loop(range(1000000, 10000001, 1000000))


# array_1 = np.random.randint(0, 100, size=(10000, 10000))
# array_2 = np.random.randint(0, 100, size=(10000, 10000))
# add_two_arrays(array_1, array_2)
