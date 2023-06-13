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

class Connection(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Connect to a database."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from a database."""

    @abstractmethod
    def query(self, query: str) -> Any:
        """Query a database."""