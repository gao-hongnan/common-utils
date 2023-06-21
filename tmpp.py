"""Decorator Functions."""
import functools
import time
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from rich.pretty import pprint


DataTypes = Union[List[int], Dict[int, int], None]
#  callable that takes any number of arguments and returns any value.
F = TypeVar("F", bound=Callable[..., Any])


# pylint: disable=invalid-name
def data_factory(data_type: str, n: int) -> DataTypes:
    """
    Generate a data structure of size n based on data_type.

    Parameters
    ----------
    data_type : str
        The type of data structure to generate. It can be 'array' for a list,
        'dict' for a dictionary, or None if no data structure is needed.
    n : int
        The size of the data structure to generate.

    Returns
    -------
    DataTypes
        The generated data structure of type list, dictionary or None.

    Raises
    ------
    ValueError
        If the data_type is not 'array', 'dict' or None.
    """
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
    """
    Decorator to compute and plot the time complexity of a function.

    Parameters
    ----------
    data_type : str
        The type of data structure that the function to be decorated uses.
        It can be 'array' for a list, 'dict' for a dictionary, or None if no
        data structure is needed.
    repeat : int, optional
        The number of times to repeat the timing test for each size of the
        data structure. The default is 1.
    plot : bool, optional
        If True, a plot of the time complexity will be displayed. The default
        is False.

    Returns
    -------
    Callable
        The decorated function that when called with a list of sizes, it
        returns a tuple containing the sizes and the average, median, best,
        and worst times over the repeats.
    """

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
                    if data_structure:
                        _ = func(data_structure, *args, **kwargs)
                    else:
                        print(n)
                        _ = func(n, *args, **kwargs)
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


@time_complexity(data_type=None, repeat=10, plot=True)
def double_for_loop(n: int) -> None:
    """Double for loop.
    n_sizes = range(100, 1001, 50)
    _ = double_for_loop(n_sizes)
    """
    for _ in range(n):
        for _ in range(n):
            pass


@time_complexity(data_type=None, repeat=10, plot=True)
def exponential(n: int) -> None:
    """Exponential.
    n_sizes = range(1, 11)
    _ = exponential(n_sizes)
    """
    _ = 2**n


@timer
def add_two_arrays(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Add two arrays together."""
    return array_1 + array_2


@time_complexity(data_type=None, repeat=10, plot=True)
def fib(n: int) -> int:
    """Calculate Fibonacci number recursively."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


n_sizes = range(10, 21)  # 100 to 1000
_ = fib(n_sizes)
