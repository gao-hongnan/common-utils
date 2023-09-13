from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type, Union, Optional, Callable

import torch
from torch.nn import CrossEntropyLoss, MSELoss

CRITERION_REGISTRY: Dict[str, Type[BaseCriterionBuilder]] = {}


@dataclass
class CriterionConfig:
    """Configuration data class for criterions."""

    name: str
    reduction: str = "mean"

    def __delattr__(self, name: str) -> None:
        if name in self.__dict__:
            super().__delattr__(name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


@dataclass
class CrossEntropyConfig(CriterionConfig):
    """CrossEntropy criterion configuration, add attributes as needed."""

    weight: Optional[torch.Tensor] = None


@dataclass
class MSELossConfig(CriterionConfig):
    """MSELoss criterion configuration, there is actually no other
    attributes to add."""


def register_criterion(
    name: str,
) -> Callable[[Type[BaseCriterionBuilder]], Type[BaseCriterionBuilder]]:
    """Decorator to register a criterion builder in the global CRITERION_REGISTRY."""

    def register_criterion_cls(
        cls: Type[BaseCriterionBuilder],
    ) -> BaseCriterionBuilder:
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Cannot register duplicate criterion {name}")
        if not issubclass(cls, BaseCriterionBuilder):
            raise ValueError(
                f"Criterion (name={name}, class={cls.__name__}) must extend BaseCriterionBuilder"
            )
        CRITERION_REGISTRY[name] = cls
        return cls

    return register_criterion_cls


class BaseCriterionBuilder:
    """Base class for criterion builders."""

    def __init__(self, config: CriterionConfig):
        self.config = config

    def build(self) -> torch.nn._Loss:
        """Abstract method to build a criterion."""
        raise NotImplementedError


@register_criterion("cross_entropy")
class CrossEntropyBuilder(BaseCriterionBuilder):
    def build(self) -> CrossEntropyLoss:
        return CrossEntropyLoss(**self.config.__dict__)


@register_criterion("mse_loss")
class MSEBuilder(BaseCriterionBuilder):
    def build(self) -> MSELoss:
        return MSELoss(**self.config.__dict__)


def build_criterion(config: CriterionConfig) -> Any:
    """Function to build a criterion based on provided configuration."""
    criterion_name = config.name
    criterion_builder_cls = CRITERION_REGISTRY.get(criterion_name)
    if not criterion_builder_cls:
        raise ValueError(
            f"The criterion {criterion_name} is not registered in registry."
            f"Our registry has {CRITERION_REGISTRY.keys()}"
        )
    del config.name
    criterion_builder = criterion_builder_cls(config)
    return criterion_builder.build()
