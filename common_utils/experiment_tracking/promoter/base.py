from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class ModelVersion:
    """
    A class representing a version of an ML model.

    This class follows the Value Object pattern, as each instance represents
    a version of a model and its associated properties without a specific
    identity.

    NOTE: Be aware of potential namespace conflicts with Mlflow's
    ModelVersion class (ModelVersion).

    Single Responsibility Principle is followed, as the class solely
    encapsulates the logic related to a model version.

    Attributes
    ----------
    version : int
        Version number, usually incremental.
    metrics : Dict[str, float]
        Performance metrics as a dictionary, e.g., {'accuracy': 0.95}.
    stage : str
        Stage of the model version, e.g., 'production'.
    """

    version: int
    metrics: Dict[str, float]
    stage: Optional[str] = None


class PromotionManager(ABC):
    """
    A base class (interface) for a model promotion manager.

    This class outlines the basic methods any concrete promotion manager should implement.
    """

    @abstractmethod
    def _check_if_there_exists_production_model(self) -> Literal[True, False]:
        ...

    @abstractmethod
    def _get_all_production_models(self) -> List[ModelVersion]:
        ...

    @abstractmethod
    def _get_latest_production_model(self) -> ModelVersion:
        ...

    @abstractmethod
    def _find_best_model_for_production(self, metric_name: str) -> ModelVersion:
        ...

    @abstractmethod
    def _compare_models(
        self,
        model1: ModelVersion,
        model2: ModelVersion,
        metric_name: str,
    ) -> Literal[True, False]:
        ...

    @abstractmethod
    def _transition_model_to_production(self, model_version: int) -> None:
        ...

    @abstractmethod
    def promote_to_production(self, metric_name: str) -> None:
        """The main method to promote a model to production."""
