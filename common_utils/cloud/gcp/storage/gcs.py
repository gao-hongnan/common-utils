from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from rich.logging import RichHandler
import logging
from google.cloud import storage

from common_utils.cloud.base import GCPConnector

# Setup logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")


@dataclass
class GCS(GCPConnector):
    bucket_name: str
    storage_client: storage.Client = field(init=False, repr=False)
    bucket: storage.Bucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.storage_client = storage.Client(
            credentials=self.credentials, project=self.project_id
        )
        self._init_bucket(self.bucket_name)
        self.bucket = self.storage_client.create_bucket(self.bucket_name)

    def _init_bucket(self, bucket_name: str) -> None:
        """
        Initialize a GCS bucket.

        Parameters
        ----------
        bucket_name : str
            The name of the GCS bucket.

        Returns
        -------
        None
        """
        self.bucket = self.storage_client.bucket(bucket_name)

    def create_bucket(self) -> None:
        """Creates a new GCS bucket if it doesn't exist."""
        try:
            self.storage_client.get_bucket(self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} already exists")
        except NotFound:
            self.bucket = self.storage_client.create_bucket(self.bucket_name)
            logger.info(f"Created bucket {self.bucket_name}")

    def list_gcs_files(self, prefix: str = "", **kwargs: Dict[str, Any]) -> List[str]:
        """
        List the files in a GCS bucket with an optional prefix.

        Parameters
        ----------
        bucket_name : str
            The name of the GCS bucket.
        prefix : str, optional, default: ""
            The prefix to filter files in the bucket, by default "".
        **kwargs : Dict[str, Any]
            Additional arguments to pass to the list_blobs method.

        Returns
        -------
        gcs_files: List[str]
            The list of file names in the specified GCS bucket.
        """
        blobs = self.storage_client.list_blobs(
            self.bucket_name, prefix=prefix, **kwargs
        )
        gcs_files = [blob.name for blob in blobs]
        return gcs_files

    def upload_blob(
        self,
        source_file_name: str,
        destination_blob_name: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Uploads a file to a GCS bucket.
        https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-client-libraries

        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        Parameters
        ----------
        bucket_name : str
            The ID of your GCS bucket.
        source_file_name : str
            The path to your file to upload.
        destination_blob_name : str
            The ID of your GCS object.

        Returns
        -------
        None
        """
        blob: storage.Blob = self.bucket.blob(destination_blob_name)

        # Optional: set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to upload is aborted if the object's
        # generation number does not match your precondition. For a destination
        # object that does not yet exist, set the if_generation_match precondition to 0.
        # If the destination object already exists in your bucket, set instead a
        # generation-match precondition using its generation number.
        generation_match_precondition = 0

        blob.upload_from_filename(
            source_file_name,
            if_generation_match=generation_match_precondition,
            **kwargs,
        )

    def upload_directory(
        self, source_dir: str, destination_dir: str, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Uploads a directory to a GCS bucket.
        https://cloud.google.com/storage/docs/uploading-objects
        """
        for file_path in Path(source_dir).glob("**/*"):
            if file_path.is_file():
                destination_blob_name = (
                    destination_dir + "/" + str(file_path.relative_to(source_dir))
                )
                self.upload_blob(str(file_path), destination_blob_name, **kwargs)
