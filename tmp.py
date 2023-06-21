from typing import Callable, Any, Tuple
import time
import matplotlib.pyplot as plt
import numpy as np
from common_utils.core.common import seed_all

seed_all(0)


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
            avg_times = []
            median_times = []
            best_times = []
            worst_times = []

            for n in n_sizes:
                runtimes = []
                for _ in range(repeat):
                    t1 = time.perf_counter()
                    func(n, *args, **kwargs)
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
def list_access(n: int) -> None:
    example_list = [i for i in range(n)]
    for _ in range(n):
        _ = example_list[_]


@timeit(repeat=10, plot=True)
def list_append(n: int) -> None:
    example_list = []
    for i in range(n):
        example_list.append(i)


@timeit(repeat=10, plot=True)
def list_pop(n: int) -> None:
    example_list = [i for i in range(n)]
    for _ in range(n):
        example_list.pop()


@timeit(repeat=10, plot=True)
def dict_access(n: int) -> None:
    example_dict = {i: i for i in range(n)}
    for i in range(n):
        _ = example_dict[i]


@timeit(repeat=10, plot=True)
def dict_set(n: int) -> None:
    example_dict = {}
    for i in range(n):
        example_dict[i] = i


list_access(range(1000, 10001, 1000))
list_append(range(1000, 10001, 1000))
list_pop(range(1000, 10001, 1000))
dict_access(range(1000, 10001, 1000))
dict_set(range(1000, 10001, 1000))
