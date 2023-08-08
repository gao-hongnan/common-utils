import os
import subprocess
from typing import Optional, Tuple, Dict, Union

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


# def _execute_command(self, command: str, env: Optional[Dict] = None) -> str:
#     if env is None:
#         env = dict(os.environ, AWS_PAGER="")
#     try:
#         output = subprocess.check_output(command, shell=True, env=env, stderr=subprocess.STDOUT)
#         return output.decode('utf-8').strip()
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error: {e.output.decode('utf-8').strip()}")
#         raise


class AWSManagerBase:
    """Base class for AWS managers."""

    def __init__(self, region: str) -> None:
        self.region = region

    def _execute_command(self, command: str, env: Optional[Dict] = None) -> None:
        if env is None:
            env = dict(os.environ, AWS_PAGER="")
        subprocess.check_call(command, shell=True, env=env)


if __name__ == "__main__":
    # Using it for EC2 commands
    ec2_builder = AWSCommandBuilder("aws ec2 describe-instances")
    command = ec2_builder.add_option(
        "--query", "Reservations[*].Instances[*].InstanceId"
    ).build()
    print(
        command
    )  # Outputs: aws ec2 describe-instances --query Reservations[*].Instances[*].InstanceId

    # Using it for Lambda commands
    lambda_builder = AWSCommandBuilder("aws lambda list-functions")
    command = lambda_builder.add_option("--max-items", "50").build()
    print(command)  # Outputs: aws lambda list-functions --max-items 50
