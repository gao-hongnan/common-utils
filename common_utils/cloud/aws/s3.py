import subprocess
from typing import List, Optional, Tuple, Dict
import os
from common_utils.cloud.aws.base import AWSCommandBuilder, AWSManagerBase


class S3BucketManager(AWSManagerBase):
    def __init__(self, region: str):
        self.region = region

    def create_bucket(
        self,
        base_name: str,
        bucket_type: str,
        options: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
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
            return bucket_name
        except subprocess.CalledProcessError as e:
            print(f"Failed to create bucket {bucket_name}. Error: {e}")
            raise

    def bucket_exists(self, bucket_name: str) -> bool:
        command = f"aws s3api head-bucket --bucket {bucket_name}"
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def upload_to_bucket(
        self, bucket_name: str, file_path: str, object_key: str
    ) -> None:
        command = f"aws s3 cp {file_path} s3://{bucket_name}/{object_key}"
        self._execute_command(command)

    def empty_bucket(self, bucket_name: str) -> None:
        # List and delete all objects
        command = f"aws s3 rm s3://{bucket_name} --recursive"
        self._execute_command(command)

    def delete_bucket(self, bucket_name: str, options) -> None:
        command = f"aws s3api delete-bucket --bucket {bucket_name}"
        builder = AWSCommandBuilder(command)
        for option, value in options:
            builder.add_option(option, value)
        self._execute_command(builder.build())


# Usage:
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
    "requirements.txt",
)
manager.empty_bucket(bucket)
manager.delete_bucket(bucket, options=[("--region", "us-west-2")])
