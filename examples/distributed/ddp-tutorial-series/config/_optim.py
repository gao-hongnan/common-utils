"""
This module employs the Builder and Registry design patterns to abstract
the creation of PyTorch optimizers and their configurations.

1. The **Builder Design Pattern** is being used for creating the optimizer instances.
This abstracts the process of creating an optimizer, allowing you to decouple
the construction of complex objects from their representation.

2. The **Registry Design Pattern** (or sometimes called the **Registration Design Pattern**)
is used to register optimizer builders globally. This allows for decoupled and dynamic
instantiation of optimizers based on a provided name, rather than direct instantiation.

The Builder pattern decouples the process of constructing complex objects
(optimizer and its configuration) from their representation. This makes
it easier to handle varying configurations and optimizers without changing
the core construction logic.

The Registry pattern allows dynamic instantiation of optimizers based on
provided names, thus fostering flexibility and extensibility. This is particularly
useful when adding new optimizers without modifying the core logic.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type

import torch
from torch.optim import SGD, Adam

OPTIMIZER_REGISTRY: Dict[str, Type[BaseOptimizerBuilder]] = {}


@dataclass
class OptimizerConfig:
    """
    Configuration data class for optimizers.
    Used as Dependency Injection and as a Service Locator to
    instantiate optimizers.

    Attributes
    ----------
    name : str
        The name identifier of the optimizer.
    lr : float
        Learning rate for the optimizer.
    """

    name: str
    lr: float

    def __delattr__(self, name: str) -> None:
        if name in self.__dict__:
            super().__delattr__(name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


@dataclass
class SGDConfig(OptimizerConfig):
    """SGD optimizer configuration, add attributes as needed."""

    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0


@dataclass
class AdamConfig(OptimizerConfig):
    """Adam optimizer configuration, add attributes as needed."""

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


def register_optimizer(name: str) -> Any:
    """
    Decorator to register an optimizer builder in the global OPTIMIZER_REGISTRY.

    The Registry pattern provides a systematic way to register and later instantiate
    classes based on names, promoting flexibility in adding new classes.

    Parameters
    ----------
    name : str
        Name identifier for the optimizer.

    Returns
    -------
    Callable
        Decorated class.
    """

    def register_optimizer_cls(
        cls: Type[BaseOptimizerBuilder],
    ) -> Type[BaseOptimizerBuilder]:
        if name in OPTIMIZER_REGISTRY:
            raise ValueError(f"Cannot register duplicate optimizer {name}")
        if not issubclass(cls, BaseOptimizerBuilder):
            raise ValueError(
                f"Optimizer (name={name}, class={cls.__name__}) must extend BaseOptimizerBuilder"
            )
        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return register_optimizer_cls


class BaseOptimizerBuilder:
    """
    Base class for optimizer builders utilizing the Builder design pattern.

    The Builder pattern abstracts away the complex construction process, thus
    simplifying object creation, especially for objects with many configurations.

    Attributes
    ----------
    config : OptimizerConfig
        Configuration for the optimizer.
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config

    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Abstract method to build an optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The model for which the optimizer will be built.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer instance.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in derived classes.
        """
        raise NotImplementedError


@register_optimizer("sgd")
class SGDBuilder(BaseOptimizerBuilder):
    def build(self, model: torch.nn.Module) -> Adam:
        """
        Constructs and returns a PyTorch Adam optimizer for a given model.

        Parameters
        ----------
        model : torch.nn.Module
            The model for which the optimizer will be built.

        Returns
        -------
        Adam
            The constructed Adam optimizer instance from PyTorch.
        """
        return SGD(model.parameters(), **self.config.__dict__)


@register_optimizer("adam")
class AdamBuilder(BaseOptimizerBuilder):
    """
    Builder class for PyTorch's Adam optimizer.

    This class uses the **Builder** design pattern to provide a unified interface
    for constructing different optimizers, including built-in ones from frameworks
    like PyTorch. It helps in maintaining a consistent method for building optimizers
    irrespective of their source.
    """

    def build(self, model: torch.nn.Module) -> Adam:
        """
        Constructs and returns a PyTorch Adam optimizer for a given model.

        Parameters
        ----------
        model : torch.nn.Module
            The model for which the optimizer will be built.

        Returns
        -------
        Adam
            The constructed Adam optimizer instance from PyTorch.
        """
        return Adam(model.parameters(), **self.config.__dict__)


def build_optimizer(
    config: OptimizerConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """
    Function to build an optimizer based on provided configuration, employing the
    Builder and Registry design patterns.

    The combination of these patterns enables dynamic, flexible, and extensible
    optimizer instantiation without modifying the core logic.

    Parameters
    ----------
    config : Union[OptimizerConfig, ClipLionConfig, AdaLRLionConfig]
        Configuration data class instance for the optimizer.
    model : Any
        The model for which the optimizer will be built.

    Returns
    -------
    Any
        The optimizer instance.

    Raises
    ------
    ValueError
        If the optimizer's name from the config is not found in the registry.
    """

    optimizer_name = config.name
    optimizer_builder_cls = OPTIMIZER_REGISTRY.get(optimizer_name)

    if not optimizer_builder_cls:
        raise ValueError(
            f"The optimizer {optimizer_name} is not registered in registry"
        )

    del config.name  # remove the name attribute from the config

    optimizer_builder = optimizer_builder_cls(config)
    return optimizer_builder.build(model)
