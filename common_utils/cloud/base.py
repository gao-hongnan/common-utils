from typing import Any, Dict, Optional

from google.oauth2 import service_account


# pylint: disable=too-few-public-methods
class GCPConnector:
    """
    A class to handle connections and operations on Google Cloud Platform
    (BigQuery and GCS).

    Attributes
    ----------
    project_id : str
        The project ID associated with the GCP services.
    service_account_key_json : str
        The path to the service account key JSON file.
    """

    def __init__(
        self,
        project_id: str,
        google_application_credentials: str,
        bucket_name: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Parameters
        ----------
        project_id : str
            The project ID associated with the GCP services.
        google_application_credentials : str
            The path to the service account key JSON file.
        """
        self.project_id = project_id
        self.credentials = service_account.Credentials.from_service_account_file(
            google_application_credentials
        )
        self.bucket_name = bucket_name
        self.kwargs = kwargs
