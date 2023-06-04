from pathlib import Path
from typing import Optional, List

from google.cloud import storage

from common_utils.cloud.base import GCPConnector


class GCS(GCPConnector):
    def __init__(
        self,
        project_id: str,
        google_application_credentials: str,
        bucket_name: Optional[str] = None,
    ) -> None:
        super().__init__(project_id, google_application_credentials, bucket_name)
        self.storage_client = storage.Client(
            credentials=self.credentials, project=project_id
        )
        self._init_bucket(bucket_name)

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

    def list_gcs_files(self, prefix: str = "") -> List[str]:
        """
        List the files in a GCS bucket with an optional prefix.

        Parameters
        ----------
        bucket_name : str
            The name of the GCS bucket.
        prefix : str, optional, default: ""
            The prefix to filter files in the bucket, by default "".

        Returns
        -------
        gcs_files: List[str]
            The list of file names in the specified GCS bucket.
        """
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        gcs_files = [blob.name for blob in blobs]
        return gcs_files

    def upload_blob(self, source_file_name: str, destination_blob_name: str) -> None:
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
        blob = self.bucket.blob(destination_blob_name)

        # Optional: set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to upload is aborted if the object's
        # generation number does not match your precondition. For a destination
        # object that does not yet exist, set the if_generation_match precondition to 0.
        # If the destination object already exists in your bucket, set instead a
        # generation-match precondition using its generation number.
        generation_match_precondition = 0

        blob.upload_from_filename(
            source_file_name, if_generation_match=generation_match_precondition
        )

        # print(f"File {source_file_name} uploaded to {destination_blob_name}.")

    def upload_directory(self, source_dir: str, destination_dir: str) -> None:
        """
        Uploads a directory to a GCS bucket.
        https://cloud.google.com/storage/docs/uploading-objects
        """
        for file_path in Path(source_dir).glob("**/*"):
            if file_path.is_file():
                destination_blob_name = (
                    destination_dir + "/" + str(file_path.relative_to(source_dir))
                )
                self.upload_blob(str(file_path), destination_blob_name)
