import json
import os
import subprocess
from typing import Dict, Optional, Tuple, Union

from common_utils.core.logger import Logger

# Setup logging
LOGGER = Logger(
    module_name=__name__, propagate=False, log_root_dir=None, log_file=None
).logger


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


if __name__ == "__main__":
    # Using it for EC2 commands
    create_instance_options = [
        ("--key-name", "oregon-gaohn"),
        (
            "--tag-specifications",
            'ResourceType=instance,Tags=[{"Key":"Name","Value":"GAOHN-EC2-Instance"}]',
        ),
    ]
    ec2_builder = AWSCommandBuilder("aws ec2 run-instances")
    ec2_builder.add_option("--instance-type", "t2.micro")
    ec2_builder.add_option("--image-id", "ami-00970f57473724c10")

    for option in create_instance_options:
        ec2_builder.add_option(*option)

    command = ec2_builder.build()
    print(command)  # Just to debug and see the full command

    # Using it for Lambda commands
    lambda_builder = AWSCommandBuilder("aws lambda list-functions")
    command = lambda_builder.add_option("--max-items", "50").build()
    print(command)  # Outputs: aws lambda list-functions --max-items 50
