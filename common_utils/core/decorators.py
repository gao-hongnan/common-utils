"""Decorator Functions."""
import functools
import os
import threading
import time
from typing import Any, Callable, TypeVar

import numpy as np
import psutil
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


def record_memory_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        mem_info_before = process.memory_info()
        initial_memory = mem_info_before.rss

        result = func(*args, **kwargs)

        # Get final memory usage
        mem_info_after = process.memory_info()
        final_memory = mem_info_after.rss

        memory_used = final_memory - initial_memory

        table = PrettyTable()
        table.field_names = ["Function Name", "Bytes", "Megabytes", "Gigabytes"]
        table.add_row(
            [
                func.__name__,
                f"{memory_used}",
                f"{memory_used / 1024 / 1024}",
                f"{memory_used / 1024 / 1024 / 1024}",
            ]
        )
        pprint(table)

        return result

    return wrapper


class MemoryMonitor:
    def __init__(self, interval=1):
        self.interval = interval  # Time interval in seconds between each check
        self.keep_monitoring = True

    def monitor_memory(self):
        process = psutil.Process(os.getpid())
        while self.keep_monitoring:
            mem_info = process.memory_info()
            print(f"Current memory usage: {mem_info.rss / 1024 / 1024} MB")
            time.sleep(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self.monitor_memory)
        self.keep_monitoring = True
        self.thread.start()

    def stop(self):
        self.keep_monitoring = False
        self.thread.join()


def monitor_memory_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        monitor.start()

        result = func(*args, **kwargs)

        monitor.stop()

        return result

    return wrapper


@timer
def add_two_arrays(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Add two arrays together."""
    return array_1 + array_2


@record_memory_usage
@monitor_memory_usage
def increase_memory_usage():
    data = []
    for _ in range(100000):
        data.append("x" * 1000000)  # Increase memory usage by 1 MB each iteration
        # time.sleep(0.1)  # Sleep for a bit to slow down the loop


if __name__ == "__main__":
    array_1 = np.random.randint(0, 100, size=(10000, 10000))
    array_2 = np.random.randint(0, 100, size=(10000, 10000))
    add_two_arrays(array_1, array_2)

    increase_memory_usage()
