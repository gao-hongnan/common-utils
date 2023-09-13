from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Type, Dict, Tuple
import torch
from torch.optim import Adam, SGD


OPTIMIZER_REGISTRY: Dict[str, Type[BaseOptimizerBuilder]] = {}


@dataclass
class OptimizerConfig:
    """
    Configuration data class for optimizers.

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
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0


@dataclass
class AdamConfig(OptimizerConfig):
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


OPTIMIZER_NAME_TO_CONFIG_MAPPING: Dict[str, Type[OptimizerConfig]] = {
    "adam": AdamConfig,
    "sgd": SGDConfig,
}


def register_optimizer(name: str) -> Any:
    """
    Decorator to register an optimizer builder in the global OPTIMIZER_REGISTRY.

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
    Base class for all optimizer builders.

    Attributes
    ----------
    cfg : OptimizerConfig
        Configuration for the optimizer.
    """

    def __init__(self, cfg: OptimizerConfig):
        self.cfg = cfg

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
        return SGD(model.parameters(), **self.cfg.__dict__)


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
        return Adam(model.parameters(), **self.cfg.__dict__)


def build_optimizer(
    cfg: OptimizerConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """
    Function to build an optimizer based on provided configuration.

    Parameters
    ----------
    cfg : Union[OptimizerConfig, ClipLionConfig, AdaLRLionConfig]
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
        If the optimizer's name from the cfg is not found in the registry.
    """

    optimizer_name = cfg.name
    optimizer_builder_cls = OPTIMIZER_REGISTRY.get(optimizer_name)

    if not optimizer_builder_cls:
        raise ValueError(
            f"The optimizer {optimizer_name} is not registered in registry"
        )

    del cfg.name  # remove the name attribute from the cfg

    optimizer_builder = optimizer_builder_cls(cfg)
    return optimizer_builder.build(model)
