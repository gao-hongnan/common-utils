from abc import ABC, abstractmethod
from typing import Any, Dict


class DictPersistence(ABC):
    @abstractmethod
    def save_as_dict(
        self, data: Dict[str, Any], filepath: str, **kwargs: Dict[str, Any]
    ) -> None:
        """Save a dictionary to a specific location."""

    @abstractmethod
    def load_to_dict(self, filepath: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Load a dictionary from a filepath."""
