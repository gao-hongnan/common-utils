from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelVersion:
    """
    A class representing a version of an ML model.

    This class follows the Value Object pattern, as each instance represents
    a version of a model and its associated properties without a specific
    identity.

    NOTE: Be aware of potential namespace conflicts with Mlflow's
    ModelVersion class.

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
    stage: str


class ModelClient(ABC):
    """All subclasses must implement these methods. See MLFlow Client
    as an example."""

    @abstractmethod
    def get_latest_versions(
        self, model_name: str, stages: List[str]
    ) -> List[ModelVersion]:
        """Get the latest versions of a model."""

    @abstractmethod
    def get_run(self, run_id: str) -> Dict:
        """Get a run by its ID."""

    @abstractmethod
    def search_model_versions(self, name: str):
        """Search for model versions by name."""

    @abstractmethod
    def transition_model_version_stage(
        self, name: str, version: str, stage: str
    ) -> None:
        """Transition a model version to a new stage."""


class MlflowClient(ModelClient):
    """
    Concrete class for Mlflow client. Create wrapper methods around
    MlflowClient methods.
    """

    # Implement other necessary methods...


class WandbClient(ModelClient):
    """
    Concrete class for Wandb client. Create wrapper methods around
    WandbClient methods.
    """
