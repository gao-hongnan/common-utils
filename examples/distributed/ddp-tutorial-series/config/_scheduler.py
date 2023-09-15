from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR

LR_SCHEDULER_REGISTRY: Dict[str, Type[BaseSchedulerBuilder]] = {}


@dataclass
class SchedulerConfig:
    """Base Configuration data class for learning rate schedulers."""

    optimizer: torch.optim.Optimizer
    name: str

    def __delattr__(self, name: str) -> None:
        if name in self.__dict__:
            super().__delattr__(name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


@dataclass
class ConstantLRConfig(SchedulerConfig):
    factor: float = 1  # no change
    total_iters: int = 0
    last_epoch: int = -1


@dataclass
class CosineAnnealingLRConfig(SchedulerConfig):
    T_max: int = 10
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False


def register_scheduler(
    name: str,
) -> Callable[[Type[BaseSchedulerBuilder]], Type[BaseSchedulerBuilder]]:
    """Decorator to register an LR scheduler builder in the global LR_SCHEDULER_REGISTRY."""

    def register_scheduler_cls(cls: Type[BaseSchedulerBuilder]) -> BaseSchedulerBuilder:
        if name in LR_SCHEDULER_REGISTRY:
            raise ValueError(f"Cannot register duplicate scheduler {name}")
        if not issubclass(cls, BaseSchedulerBuilder):
            raise ValueError(
                f"Scheduler (name={name}, class={cls.__name__}) must extend BaseSchedulerBuilder"
            )
        LR_SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_scheduler_cls


class BaseSchedulerBuilder:
    """Base class for LR scheduler builders."""

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def build(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Abstract method to build a LR scheduler."""
        raise NotImplementedError


@register_scheduler("constant_lr")
class ConstantLRBuilder(BaseSchedulerBuilder):
    """Builder for constant LR scheduler."""

    def __init__(self, config: ConstantLRConfig):
        super().__init__(config)

    def build(self) -> torch.optim.lr_scheduler._LRScheduler:
        return ConstantLR(self.config.optimizer, self.config.factor)


@register_scheduler("cosine_annealing_lr")
class CosineAnnealingLRBuilder(BaseSchedulerBuilder):
    """Builder for cosine annealing LR scheduler."""

    def __init__(self, config: CosineAnnealingLRConfig):
        super().__init__(config)

    def build(self) -> torch.optim.lr_scheduler._LRScheduler:
        return CosineAnnealingLR(self.config.optimizer, **self.config.__dict__)


def build_scheduler(config: SchedulerConfig) -> Any:
    """Function to build an LR scheduler based on provided configuration."""
    scheduler_name = config.name
    scheduler_builder_cls = LR_SCHEDULER_REGISTRY.get(scheduler_name)
    if not scheduler_builder_cls:
        raise ValueError(
            f"The scheduler {scheduler_name} is not registered in the registry."
            f"Our registry has {LR_SCHEDULER_REGISTRY.keys()}"
        )
    del config.name
    scheduler_builder = scheduler_builder_cls(config)
    return scheduler_builder.build()
