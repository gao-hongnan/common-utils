import subprocess
from typing import List, Optional, Tuple

from common_utils.cloud.aws.base import LOGGER, AWSCommandBuilder, AWSManagerBase


class S3BucketManager(AWSManagerBase):
    """Manager class for AWS S3 Buckets.

    Provides utilities to create, check, upload to, and delete S3 buckets.
    """

    def create_bucket(
        self,
        base_name: str,
        bucket_type: str,
        options: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Create an S3 bucket with a given name and type.

        Parameters
        ----------
        base_name : str
            Base name for the bucket.
        bucket_type : str
            Type of the bucket.
        options : List[Tuple[str, str]], optional
            Additional AWS options. Defaults to None.

        Returns
        -------
        str
            Name of the created bucket.

        Raises
        ------
        subprocess.CalledProcessError
            If the bucket creation command returns a non-zero exit status.
        """
        bucket_name = f"{base_name}-{bucket_type}"

        if self.bucket_exists(bucket_name):
            print(f"Bucket {bucket_name} already exists.")
            return bucket_name

        builder = (
            AWSCommandBuilder("aws s3api create-bucket")
            .add_option("--bucket", bucket_name)
            .add_option(
                "--create-bucket-configuration", f"LocationConstraint={self.region}"
            )
        )

        if options:
            for option, value in options:
                builder.add_option(option, value)

        try:
            self._execute_command(builder.build())
            LOGGER.info(f"Created bucket: {bucket_name}.")
            return bucket_name
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to create bucket: {bucket_name}. Error: {e}")
            raise

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if an S3 bucket exists.

        Parameters
        ----------
        bucket_name : str
            The name of the bucket.

        Returns
        -------
        bool
            True if the bucket exists, False otherwise.
        """
        command = f"aws s3api head-bucket --bucket {bucket_name}"
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
            return True
        except subprocess.CalledProcessError as e:
            LOGGER.warning(e.output.decode("utf-8").strip())
            return False

    def upload_to_bucket(
        self, bucket_name: str, file_path: str, object_key: str
    ) -> None:
        """Upload a file to an S3 bucket.

        Parameters
        ----------
        bucket_name : str
            The name of the destination bucket.
        file_path : str
            Path to the file to upload.
        object_key : str
            Key for the object in the S3 bucket.
        """
        command = f"aws s3 cp {file_path} s3://{bucket_name}/{object_key}"
        try:
            self._execute_command(command)
            LOGGER.info(f"Uploaded {file_path} to {bucket_name}/{object_key}.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to upload {file_path} to {bucket_name}. Error: {e}")
            raise

    def empty_bucket(self, bucket_name: str) -> None:
        """Empty an S3 bucket.

        Parameters
        ----------
        bucket_name : str
            The name of the bucket to empty.
        """
        command = f"aws s3 rm s3://{bucket_name} --recursive"
        try:
            self._execute_command(command)
            LOGGER.info(f"Emptied the bucket: {bucket_name}.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to empty the bucket {bucket_name}. Error: {e}")
            raise

    def delete_bucket(self, bucket_name: str, options: List[Tuple[str, str]]) -> None:
        """Delete an S3 bucket.

        Parameters
        ----------
        bucket_name : str
            The name of the bucket to delete.
        options : List[Tuple[str, str]]
            Additional AWS options.
        """
        command = f"aws s3api delete-bucket --bucket {bucket_name}"
        builder = AWSCommandBuilder(command)
        for option, value in options:
            builder.add_option(option, value)
        try:
            self._execute_command(builder.build())
            LOGGER.info(f"Deleted the bucket: {bucket_name}.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to delete the bucket {bucket_name}. Error: {e}")
            raise


if __name__ == "__main__":
    manager = S3BucketManager(region="us-west-2")
    create_bucket_flags = [
        ("--no-object-lock-enabled-for-bucket", None),
    ]
    bucket = manager.create_bucket(
        base_name="gaohn-oregon-test-demo",
        bucket_type="testtest",
        options=create_bucket_flags,
    )
    manager.upload_to_bucket(
        bucket,
        "/Users/reighns/gaohn/pipeline/common-utils/requirements.txt",
        "project/reighns/requirements.txt",
    )
    manager.empty_bucket(bucket)
    manager.delete_bucket(bucket, options=[("--region", "us-west-2")])

    # bucket_name = "gaohn-oregon-test-demo-common"  #  f"{base_name}-{bucket_type}"
    # manager.upload_to_bucket(
    #     bucket_name,
    #     "~/Downloads/llm-foundry-90795f37c16c008aae954df55fc4f3323bc581e4.zip",
    #     "source_files/llm-foundry-90795f37c16c008aae954df55fc4f3323bc581e4.zip",
    # )
