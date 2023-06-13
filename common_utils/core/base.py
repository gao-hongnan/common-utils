from abc import ABC, abstractmethod
from typing import Any, Dict


class DictPersistence(ABC):
    """Abstract class for saving and loading dictionary."""

    @abstractmethod
    def save_as_dict(
        self, data: Dict[str, Any], filepath: str, **kwargs: Dict[str, Any]
    ) -> None:
        """Save a dictionary to a specific location."""

    @abstractmethod
    def load_to_dict(self, filepath: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Load a dictionary from a filepath."""


class Connection(ABC):
    """Abstract class for database connection."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to a database."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from a database."""

    @abstractmethod
    def query(self, query: str) -> Any:
        """Query a database."""


class Storage(ABC):
    """Abstract class for storage."""

    @abstractmethod
    def save_file(self) -> None:
        """Save a file to a specific location."""

    @abstractmethod
    def save_files(self) -> None:
        """Save multiple files to a specific location."""

    @abstractmethod
    def load_file(self) -> None:
        """Load a file from a specific location."""

    @abstractmethod
    def load_files(self) -> None:
        """Load multiple files from a specific location."""
