import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Tuple


def fib(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


def measure_time(
    func: Callable[..., int], n_values: List[int], repeat: int = 1, plot: bool = True
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    avg_times = []
    median_times = []
    best_times = []
    worst_times = []

    for n in n_values:
        runtimes = []
        for _ in range(repeat):
            start_time = time.perf_counter()
            func(n)
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)

        avg_times.append(np.mean(runtimes))
        median_times.append(np.median(runtimes))
        best_times.append(np.min(runtimes))
        worst_times.append(np.max(runtimes))

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, avg_times, "o-", label="Average")
        plt.plot(n_values, median_times, "o-", label="Median")
        plt.plot(n_values, best_times, "o-", label="Best")
        plt.plot(n_values, worst_times, "o-", label="Worst")
        plt.xlabel("Input Size (n)")
        plt.ylabel("Execution Time (s)")
        plt.legend()
        plt.grid(True)
        plt.title(f"Time Complexity of {func.__name__}")
        plt.show()

    return n_values, avg_times, median_times, best_times, worst_times


n_values = list(range(1, 31))  # Adjust this based on the complexity of your function
measure_time(fib, n_values, repeat=3, plot=True)
