import subprocess
from typing import List, Optional, Tuple

from common_utils.cloud.aws.base import AWSCommandBuilder, AWSManagerBase, LOGGER


class EC2InstanceManager(AWSManagerBase):
    """Manager class for AWS EC2 Instances.

    Provides utilities to create, check status, start, stop, and terminate EC2 instances.
    """

    def create_instance(
        self,
        instance_type: str,
        ami_id: str,
        options: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Create an EC2 instance.

        Parameters
        ----------
        instance_type : str
            The type of EC2 instance.
        ami_id : str
            The AMI ID to launch the instance with.
        options : List[Tuple[str, str]], optional
            Additional AWS options. Defaults to None.

        Returns
        -------
        str
            ID of the created instance.
        """
        builder = (
            AWSCommandBuilder("aws ec2 run-instances")
            .add_option("--instance-type", instance_type)
            .add_option("--image-id", ami_id)
        )

        if options:
            for option, value in options:
                builder.add_option(option, value)

        try:
            full_command = builder.build()
            print(full_command)
            output = self._execute_command(builder.build())
            instance_id = output.get("Instances", [{}])[0].get("InstanceId")
            LOGGER.info(f"Created EC2 instance: {instance_id}.")
            return instance_id
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to create EC2 instance. Error: {e}")
            raise

    def is_instance_running(self, instance_id: str) -> bool:
        """Check if an EC2 instance is running.

        Parameters
        ----------
        instance_id : str
            The ID of the EC2 instance.

        Returns
        -------
        bool
            True if the instance is running, False otherwise.
        """
        command = f"aws ec2 describe-instances --instance-id {instance_id}"
        try:
            output = self._execute_command(command)
            state = (
                output.get("Reservations", [{}])[0]
                .get("Instances", [{}])[0]
                .get("State", {})
                .get("Name")
            )
            return state == "running"
        except subprocess.CalledProcessError as e:
            LOGGER.warning(e.output.decode("utf-8").strip())
            return False

    def start_instance(self, instance_id: str) -> None:
        """Start an EC2 instance.

        Parameters
        ----------
        instance_id : str
            The ID of the EC2 instance.
        """
        command = f"aws ec2 start-instances --instance-ids {instance_id}"
        try:
            self._execute_command(command)
            LOGGER.info(f"Started EC2 instance: {instance_id}.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to start EC2 instance {instance_id}. Error: {e}")
            raise

    def stop_instance(self, instance_id: str) -> None:
        """Stop an EC2 instance.

        Parameters
        ----------
        instance_id : str
            The ID of the EC2 instance.
        """
        command = f"aws ec2 stop-instances --instance-ids {instance_id}"
        try:
            self._execute_command(command)
            LOGGER.info(f"Stopped EC2 instance: {instance_id}.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to stop EC2 instance {instance_id}. Error: {e}")
            raise

    def terminate_instance(self, instance_id: str) -> None:
        """Terminate an EC2 instance.

        Parameters
        ----------
        instance_id : str
            The ID of the EC2 instance.
        """
        command = f"aws ec2 terminate-instances --instance-ids {instance_id}"
        try:
            self._execute_command(command)
            LOGGER.info(f"Terminated EC2 instance: {instance_id}.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to terminate EC2 instance {instance_id}. Error: {e}")
            raise


if __name__ == "__main__":
    # Initialize the EC2InstanceManager
    ec2_manager = EC2InstanceManager(region="us-west-2")

    # Some parameters required for creating the instance
    instance_type = "t2.micro"  # For instance, using the t2.micro type
    ami_id = "ami-00970f57473724c10"  # Example AMI ID for Amazon Linux 2 in us-west-2; this might change over time

    # Additional optional parameters can be added as needed
    # TODO: check if we should use : or = for the options
    """
    aws ec2 \
    run-instances --count 1 \
    --image-id ami-0a5a20c6f44946afe \
    --instance-type t3.large \
    --key-name oregon-gaohn \
    --security-group-ids sg-03e74c4a82b6550dd \
    --subnet-id subnet-0ed0273457ba67b31 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Category,Value=MLFlowServer},{Key=Name,Value=MLFlowServer},{Key=parallelcluster:cluster-name,Value=gaohn-oregon-test-demo}]'
    """
    create_instance_options = [
        ("--key-name", "oregon-gaohn"),
        (
            "--tag-specifications",
            '\'{ "ResourceType": "instance", "Tags": [ {"Key":"Name", "Value":"GAOHN-EC2-Instance"} ] }\'',
        ),
    ]

    # Create the EC2 instance
    instance_id = ec2_manager.create_instance(
        instance_type=instance_type,
        ami_id=ami_id,
        options=create_instance_options,
    )

    # Check if the instance is running
    import time

    time.sleep(60)  # Wait for the instance to start
    if ec2_manager.is_instance_running(instance_id):
        LOGGER.info(f"EC2 instance {instance_id} is running.")
    else:
        LOGGER.info(f"EC2 instance {instance_id} is not running.")

    ec2_manager.terminate_instance(instance_id=instance_id)
    # You can add more code here to stop, start or terminate the instance if needed
