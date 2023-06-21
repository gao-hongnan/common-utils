from typing import Callable, Any, Tuple
import time
import matplotlib.pyplot as plt
import numpy as np
from common_utils.core.common import seed_all

seed_all(1992)


def timeit(
    repeat: int = 1, plot: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Tuple]]:
    """
    Decorator for timing a function. Calculates and plots (optional) the time
    complexity of the decorated function using average, median, best, and worst
    time taken over specified runs.

    Parameters
    ----------
    repeat : int, optional
        Number of times to repeat the test for each n, by default 1.
    plot : bool, optional
        If True, plot the time complexity graph, by default False.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Tuple]]
        The decorated function with timing.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Tuple]:
        def wrapper(n_sizes: list, *args: Any, **kwargs: Any) -> Tuple:
            print(f"n_sizes: {n_sizes}")
            avg_times = []
            median_times = []
            best_times = []
            worst_times = []

            for n in n_sizes:
                # create a list of n elements
                array: list = [i for i in range(n)]
                # note array is created outside the loop
                runtimes = []
                for _ in range(repeat):
                    t1 = time.perf_counter()
                    func(
                        n, array, *args, **kwargs
                    )  # <--- this is where it calls the function with n as argument
                    t2 = time.perf_counter()
                    runtimes.append(t2 - t1)

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


@timeit(repeat=10, plot=True)
def list_access(n: int, array) -> None:
    _ = array[n // 2]


@timeit(repeat=10, plot=True)
def list_append(n: int, array) -> None:
    array.append(n)


@timeit(repeat=10, plot=True)
def list_insert(n: int, array) -> None:
    array.insert(0, 1)


# @timeit(repeat=10, plot=True)
# def for_loop(n: int, array) -> None:
#     for i in range(n):


@timeit(repeat=10, plot=True)
def dict_access(n: int) -> None:
    example_dict = {i: i for i in range(n)}
    _ = example_dict[n // 2]


@timeit(repeat=10, plot=True)
def dict_set(n: int) -> None:
    example_dict = {}
    example_dict[n] = n


list_access(range(1000000, 10000001, 1000000))
list_append(range(1000000, 10000001, 1000000))
list_insert(range(1000000, 10000001, 1000000))
# dict_access(range(1000, 10001, 1000))
# dict_set(range(1000, 10001, 1000))
