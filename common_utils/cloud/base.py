from dataclasses import dataclass, field

from google.auth.credentials import Credentials
from google.oauth2 import service_account


# pylint: disable=too-few-public-methods
@dataclass
class GCPConnector:
    """
    A class to handle connections and operations on Google Cloud Platform
    (BigQuery and GCS).

    Attributes
    ----------
    project_id : str
        The project ID associated with the GCP services.
    google_application_credentials : str
        The path to the service account key JSON file.
    """

    # ellipsis (...) means required
    project_id: str
    google_application_credentials: str
    credentials: Credentials = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize the credentials for the GCP services.
        """
        self.credentials = service_account.Credentials.from_service_account_file(
            self.google_application_credentials
        )
