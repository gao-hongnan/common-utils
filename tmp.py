import json
import logging
import os
import subprocess
from typing import Dict, List, Optional, Tuple, Union

from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler()],
)
# Setup logging
LOGGER = logging.getLogger("rich")

OptionType = Union[Tuple[str], Tuple[str, str]]


class AWSCommandBuilder:
    """
    Constructs AWS CLI commands by chaining options.

    Attributes
    ----------
    command_parts : list of str
        List of strings representing the parts of the AWS command.

    Methods
    -------
    add_option(option: str, value: Optional[str] = None) -> 'AWSCommandBuilder':
        Add an option (and its value) to the command.
    build() -> str:
        Get the final constructed command as a string.
    """

    def __init__(self, base_command: str) -> None:
        """
        Initialize AWSCommandBuilder with the base command.

        Parameters
        ----------
        base_command : str
            The base command to initialize with, e.g., 'aws s3 ls'.
        """
        self.command_parts = [base_command]

    def add_option(
        self, option: str, value: Optional[str] = None
    ) -> "AWSCommandBuilder":
        """
        Add an option to the command.

        If the option does not have a value, the value parameter should be
        None, consequently adding the option without a value.

        Parameters
        ----------
        option : str
            The option/flag to add, e.g., '--bucket'.
        value : str, optional
            The value for the option, if any.

        Returns
        -------
        AWSCommandBuilder
            Returns the builder object to allow for method chaining.
        """
        if value:
            self.command_parts.append(f"{option} {value}")
        else:
            self.command_parts.append(option)
        return self

    def build(self) -> str:
        """
        Construct and return the final AWS CLI command.

        Example
        -------
        >>> builder = AWSCommandBuilder("aws s3api create-bucket")
        >>> builder.add_option("--bucket", "my-bucket")
        >>> builder.add_option("--create-bucket-configuration", "LocationConstraint=us-west-2")
        >>> builder.build()
        'aws s3api \
            create-bucket \
            --bucket my-bucket \
            --create-bucket-configuration LocationConstraint=us-west-2'

        Returns
        -------
        str
            The constructed AWS CLI command.
        """
        return " ".join(self.command_parts)


class AWSManagerBase:
    """Base class for AWS managers.

    This class provides basic utilities for executing AWS commands.

    Attributes
    ----------
    region : str
        AWS region for the manager.
    """

    def __init__(self, region: str) -> None:
        """Initialize the AWSManagerBase.

        Parameters
        ----------
        region : str
            AWS region for the manager.
        """
        self.region = region

    def _execute_command(
        self, command: str, env: Optional[Dict] = None
    ) -> Union[bytes, Dict[str, str]]:
        """Execute a command with a given environment.

        Parameters
        ----------
        command : str
            The command to execute.
        env : Dict, optional
            The environment variables to set for the command. Defaults to None.

        Returns
        -------
        Dict[str, str]
            The output of the command as a dictionary.

        Raises
        ------
        subprocess.CalledProcessError
            If the command returns a non-zero exit status.
        """
        if env is None:
            env = dict(os.environ, AWS_PAGER="")
        try:
            output_bytes = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
            output_str = output_bytes.decode("utf-8").strip()

            # Check if the output is JSON
            try:
                return json.loads(output_str)
            except json.JSONDecodeError:
                LOGGER.info(output_str)
                return output_str

        except subprocess.CalledProcessError as e:
            LOGGER.error(
                f"Command failed with error: {e.output.decode('utf-8').strip()}"
            )
            raise


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

    bucket_name = "gaohn-oregon-test-demo-common" # f"{base_name}-{bucket_type}"
    manager.empty_bucket(bucket_name)

    # TODO: use argparse
    delete_bucket_options = [("--region", "us-west-2")]
    manager.delete_bucket(bucket_name, options=delete_bucket_options)

