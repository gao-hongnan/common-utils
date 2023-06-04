from typing import Any, List, Tuple, Union, Dict

import pandas as pd
from google.cloud import bigquery

from common_utils.cloud.base import GCPConnector


class BigQuery(GCPConnector):
    def __init__(
        self,
        project_id: str,
        google_application_credentials: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(project_id, google_application_credentials)
        self.bigquery_client = bigquery.Client(
            credentials=self.credentials, project=project_id, **kwargs
        )

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

    def load_job_config(self, **kwargs: Dict[str, Any]) -> bigquery.LoadJobConfig:
        return bigquery.LoadJobConfig(**kwargs)

    def load_data_from_dataframe(self, dataframe, table_id, schema, mode, **kwargs):
        """
        Loads data from a DataFrame to BigQuery.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to load.
        table_id : str
            The full ID of the table where the data will be loaded.
        schema: List[SchemaField]
            The schema fields to use for the table.
        """
        job_config = self.load_job_config(schema=schema, write_disposition=mode)

        load_job = self.bigquery_client.load_table_from_dataframe(
            dataframe, table_id, job_config=job_config, **kwargs
        )

        load_job.result()  # Waits for the job to complete.

        print(
            f"Loaded {dataframe.shape[0]} rows and {dataframe.shape[1]} columns to {table_id}"
        )

    # def load_gcs_to_bq(
    #     self, gcs_uri: str, dataset_id: str, table_id: str, schema: list
    # ) -> None:
    #     """
    #     Load data from Google Cloud Storage into BigQuery.

    #     Parameters
    #     ----------
    #     gcs_uri : str
    #         The URI of the GCS file to load. It should be in the format gs://<bucket_name>/<file_path>.
    #     dataset_id : str
    #         The ID of the BigQuery dataset to load the data into.
    #     table_id : str
    #         The ID of the BigQuery table to load the data into.
    #     schema : list
    #         The schema of the BigQuery table to load the data into.
    #     """
    #     dataset_ref = self.bigquery_client.dataset(dataset_id)
    #     job_config = LoadJobConfig()
    #     job_config.source_format = SourceFormat.CSV
    #     job_config.skip_leading_rows = 1
    #     job_config.autodetect = True
    #     job_config.schema = schema

    #     load_job = self.bigquery_client.load_table_from_uri(
    #         gcs_uri, dataset_ref.table(table_id), job_config=job_config
    #     )
    #     load_job.result()


# if __name__ == "__main__":
#     # TODO: To put in pytests.

#     gcp_secrets = GCPSecrets(
#         project_id=PROJECT_ID, google_application_credentials=SERVICE_ACCOUNT_KEY_JSON
#     )

#     # Instantiate the class
#     bigquery = BigQuery(gcp_secrets.project_id, gcp_secrets.google_application_credentials)
#     gcs = GCS(gcp_secrets.project_id, gcp_secrets.google_application_credentials)

#     # Execute a query in BigQuery
#     query = """
#         SELECT *
#         FROM `gao-hongnan.imdb_dbt_filtered_movies.filtered_movies`
#         WHERE primaryTitle IS NOT NULL
#             AND originalTitle IS NOT NULL
#             AND averageRating IS NOT NULL
#             AND genres IS NOT NULL
#             AND runtimeMinutes IS NOT NULL
#             AND startYear > 2014
#         ORDER BY tconst DESC
#         LIMIT 100
#         """
#     df = bigquery.query(query)
#     df.to_csv("./data/raw/imdb_dbt_filtered_movies.csv", index=False)
#     # List files in a GCS bucket
#     bucket_name = "gaohn"
#     files = gcs.list_gcs_files(bucket_name)

#     # Print the results
#     pprint(df.head(20))
#     pprint(files)

#     pprint(df.head(20))

#     pprint(len(df))
