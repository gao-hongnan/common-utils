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
    """Abstract class for database connection.

    See common_utils/cloud/gcp/database/bigquery.py for example.

    NOTE:
        1. In particular, it should have a property or attribute table_name.
    """

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
    """Abstract class for storage.

    See common_utils/cloud/gcp/storage/gcs.py for example.

    NOTE:
        1. Subsequent cloud storages should abide by this interface. This means
            turning S3, Azure, etc. into a class that inherits from this class.
    """

    @abstractmethod
    def upload_blob(self) -> None:
        """Save a file to a specific location."""

    @abstractmethod
    def upload_blobs(self) -> None:
        """Save multiple files to a specific location."""

    @abstractmethod
    def upload_directory(self) -> None:
        """Save a directory to a specific location."""

    @abstractmethod
    def download_blob(self) -> None:
        """Load a file from a specific location."""

    @abstractmethod
    def download_blobs(self) -> None:
        """Load multiple files from a specific location."""
