from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from common_utils.cloud.base import GCPConnector
from common_utils.core.logger import Logger

# Setup logging
logger = Logger(
    module_name=__name__, propagate=False, log_root_dir=None, log_file=None
).logger


@dataclass
class BigQuery(GCPConnector):
    # usually you call project_id.dataset_id.table_name
    dataset: str  # The ID of the dataset to use.
    table_name: str  # The name of the table to use.
    _dataset_id: str = field(init=False)  # The ID of the dataset to use.
    _table_id: str = field(init=False)  # The full ID of the table to use.
    bigquery_client: bigquery.Client = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize the BigQuery client for the GCP services.
        """
        super().__post_init__()
        self.bigquery_client = bigquery.Client(
            credentials=self.credentials, project=self.project_id
        )

    @property
    def table_id(self) -> str:
        """Return the full ID of the table to use."""
        return f"{self.project_id}.{self.dataset}.{self.table_name}"

    @property
    def dataset_id(self) -> str:
        """Return the ID of the dataset to use."""
        return f"{self.project_id}.{self.dataset}"

    def check_if_dataset_exists(self) -> Literal[True, False]:
        """Check if a dataset exists."""
        try:
            self.bigquery_client.get_dataset(self.dataset_id)
            logger.info(f"Dataset {self.dataset_id} already exists")
            return True
        except NotFound:
            logger.warning(
                f"""Dataset {self.dataset_id} does not exist.
                Please create it using `create_dataset`."""
            )
            return False

    def create_dataset(self) -> None:
        """Creates a new BigQuery dataset if it doesn't exist."""
        dataset = bigquery.Dataset(self.dataset_id)
        self.bigquery_client.create_dataset(dataset)
        logger.info(f"Created dataset {self.dataset_id}")

    def check_if_table_exists(self) -> Literal[True, False]:
        """Check if a table exists."""
        try:
            self.bigquery_client.get_table(self.table_id)
            logger.info(f"Table {self.table_id} already exists")
            return True
        except NotFound:
            logger.warning(
                f"""Table {self.table_id} does not exist.
                Please create it using `create_table`."""
            )
            return False

    def create_table(self, schema: List[bigquery.SchemaField]) -> None:
        """Creates a new BigQuery table if it doesn't exist."""
        table = bigquery.Table(self.table_id, schema=schema)
        self.bigquery_client.create_table(table)
        logger.info(f"Created table {self.table_id}")

    def query(
        self, query: str, as_dataframe: bool = True
    ) -> Union[List[Tuple[Any]], pd.DataFrame]:
        """
        Execute a query in BigQuery and return the result as a DataFrame.

        Parameters
        ----------
        query : str
            The SQL query to execute in BigQuery.

        Returns
        -------
        pd.DataFrame
            The result of the query as a DataFrame.
        """
        query_job = self.bigquery_client.query(query)
        results = query_job.result()
        return results.to_dataframe() if as_dataframe else results

    def load_job_config(
        self,
        schema: Optional[List[bigquery.SchemaField]] = None,
        write_disposition: str = "WRITE_APPEND",
        **kwargs: Dict[str, Any],
    ) -> bigquery.LoadJobConfig:
        """
        Creates a load job configuration for BigQuery.
        """
        return bigquery.LoadJobConfig(
            schema=schema, write_disposition=write_disposition, **kwargs
        )

    def load_table_from_dataframe(
        self,
        df: pd.DataFrame,
        job_config: bigquery.LoadJobConfig,
        **kwargs,
    ) -> None:
        """
        Loads data from a DataFrame to BigQuery.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to load.
        table_id : str
            The full ID of the table where the data will be loaded.
        schema: List[SchemaField]
            The schema fields to use for the table. None if the table already exists.
        """
        load_job = self.bigquery_client.load_table_from_dataframe(
            df, self.table_id, job_config=job_config, **kwargs
        )

        load_job.result()  # Waits for the job to complete.

        logger.info(f"Loaded {load_job.output_rows} rows into {self.table_id}.")
