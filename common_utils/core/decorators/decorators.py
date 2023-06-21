"""Decorator Functions."""
import functools
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, TypeVar

import numpy as np
import psutil
from fastapi import Request
from prettytable import PrettyTable
from rich.pretty import pprint

#  callable that takes any number of arguments and returns any value.
F = TypeVar("F", bound=Callable[..., Any])


def construct_response(func: F) -> F:
    """Construct a JSON response for an endpoint.

    Supported Frameworks:
    - FastAPI
    To support Flask and Django.

    Reference:
    https://madewithml.com/courses/mlops/api/#decorators
    """

    @functools.wraps(func)
    def wrap(request: Request, *args: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        results = func(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url,  # ._url
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


# TODO: For memory usage, consider checking out memory_profiler.
# Coding it my own way is not good as it does not take into account
# a lot of minute details, and it does not work for multithreading and
# multiprocessing.
def record_memory_usage(func: F) -> F:
    """Memory usage decorator."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Any:
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
    def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        monitor = MemoryMonitor()
        monitor.start()

        result = func(*args, **kwargs)

        monitor.stop()

        return result

    return wrapper


@record_memory_usage
@monitor_memory_usage
def increase_memory_usage():
    data = []
    for _ in range(100000):
        data.append("x" * 1000000)  # Increase memory usage by 1 MB each iteration
        # time.sleep(0.1)  # Sleep for a bit to slow down the loop


if __name__ == "__main__":
    increase_memory_usage()
