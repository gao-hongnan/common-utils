from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

from google.cloud import storage
from google.cloud.exceptions import NotFound

from common_utils.cloud.base import GCPConnector
from common_utils.core.logger import Logger

# Setup logging
# logging.basicConfig(
#     level="INFO",
#     format="%(asctime)s [%(levelname)s]: %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     handlers=[RichHandler()],
# )

# logger = logging.getLogger("rich")


# Setup logging
logger = Logger(
    module_name=__name__, propagate=False, log_root_dir=None, log_file=None
).logger


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

    def check_if_bucket_exists(self) -> Literal[True, False]:
        """Check if a bucket exists."""
        try:
            self.storage_client.get_bucket(self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} already exists")
            return True
        except NotFound:
            logger.info(
                f"""Bucket {self.bucket_name} does not exist.
                Please create it using `create_bucket`."""
            )
            return False

    def create_bucket(self) -> None:
        """Creates a new GCS bucket if it doesn't exist."""
        self.storage_client.create_bucket(self.bucket_name)
        logger.info(f"Bucket {self.bucket_name} created")

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

    def create_blob(self, destination_blob_name: str) -> storage.Blob:
        """Creates a new blob in the bucket. See upload_blob for example."""
        blob = self.bucket.blob(destination_blob_name)
        return blob

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
        # Optional: set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to upload is aborted if the object's
        # generation number does not match your precondition. For a destination
        # object that does not yet exist, set the if_generation_match precondition to 0.
        # If the destination object already exists in your bucket, set instead a
        # generation-match precondition using its generation number.
        generation_match_precondition = 0

        blob = self.create_blob(destination_blob_name)

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

    def download_blob(
        self,
        source_blob_name: str,
        destination_file_name: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Downloads a blob from the bucket.
        https://cloud.google.com/storage/docs/downloading-objects

        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"

        # The ID of your GCS object
        # source_blob_name = "storage-object-name"

        # The path to which the file should be downloaded
        # destination_file_name = "local/path/to/file"
        """

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = self.create_blob(source_blob_name)

        blob.download_to_filename(destination_file_name, **kwargs)
