from __future__ import annotations

import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import networkx as nx


# TODO: how to do parallelism?
class Task:
    """This is a Node in the graph."""

    def __init__(self, func: Callable, pipeline: Pipeline) -> None:
        self.func = func
        self.pipeline = pipeline

        self.name: str = func.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __rshift__(self, other: Task) -> Task:
        self.pipeline.add_dependency(self, other)
        return other


class Pipeline:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.tasks = {}
        self.task_data = {}

    def task(self):
        def decorator(f):
            task = Task(f, self)
            self.tasks[task.name] = task
            self.graph.add_node(task.name)
            return task

        return decorator

    def add_dependency(self, upstream, downstream):
        self.graph.add_edge(upstream.name, downstream.name)

    def run(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            for task_name in nx.topological_sort(self.graph):
                task_func = self.tasks[task_name]
                if task_name in self.task_data:
                    args = self.task_data[task_name]
                else:
                    args = ()
                future = executor.submit(task_func, *args)
                result = future.result()
                for successor in self.graph.successors(task_name):
                    self.task_data[successor] = (result,)
            return self.task_data

    def visualize(self) -> None:
        pos = nx.spring_layout(self.graph)
        for node in self.graph.nodes:
            plt.text(
                pos[node][0],
                pos[node][1],
                node,
                fontsize=12,
                bbox=dict(
                    facecolor="blue", edgecolor="black", boxstyle="round,pad=0.2"
                ),
            )
        nx.draw(self.graph, pos, with_labels=False, node_color="w", edgecolors="k")
        plt.show()


pipeline = Pipeline()


@pipeline.task()
def feature_pipeline():
    print(f"{datetime.datetime.now()} - Running feature pipeline")
    time.sleep(2)
    print(f"{datetime.datetime.now()} - Finished feature pipeline")
    return "features"


@pipeline.task()
def training_pipeline(features):
    print(f"{datetime.datetime.now()} - Running training pipeline with {features}")
    time.sleep(2)
    print(f"{datetime.datetime.now()} - Finished training pipeline")
    return "trained_model"


@pipeline.task()
def prediction_pipeline(trained_model):
    print(
        f"{datetime.datetime.now()} - Running prediction pipeline with {trained_model}"
    )
    time.sleep(5)
    print(f"{datetime.datetime.now()} - Finished prediction pipeline")
    return "predictions"


@pipeline.task()
def evaluation_pipeline(trained_model):
    print(
        f"{datetime.datetime.now()} - Running evaluation pipeline with {trained_model}"
    )
    time.sleep(5)
    print(f"{datetime.datetime.now()} - Finished evaluation pipeline")
    return "evaluation_metrics"


@pipeline.task()
def validation_pipeline(trained_model):
    print(
        f"{datetime.datetime.now()} - Running validation pipeline with {trained_model}"
    )
    time.sleep(5)
    print(f"{datetime.datetime.now()} - Finished validation pipeline")
    return "validation_metrics"


feature_pipeline >> training_pipeline >> prediction_pipeline
feature_pipeline >> training_pipeline >> evaluation_pipeline
feature_pipeline >> training_pipeline >> validation_pipeline


# pipeline.visualize()
pipeline.run()
