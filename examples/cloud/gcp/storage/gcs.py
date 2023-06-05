import os

from dotenv import load_dotenv
import pandas as pd
import time
import math
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
import datetime
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.cloud.gcp.storage.bigquery import BigQuery
from google.cloud.exceptions import NotFound
from rich import print
from rich.pretty import pprint
import pytz
from google.cloud import bigquery
import logging


from rich.logging import RichHandler

# Setup logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

load_dotenv(dotenv_path="examples/cloud/gcp/storage/.env")

PROJECT_ID = os.getenv("PROJECT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BUCKET_NAME = os.getenv("BUCKET_NAME")
print(PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, BUCKET_NAME)

# gcs = GCS(PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, bucket_name=BUCKET_NAME)
# files = gcs.list_gcs_files()

# print(files)


btc_dict = {"binance_btcusdt_spot": "BTCUSDT"}  # tablename and symbol
eth_dict = {"binance_ethusdt_spot": "ETHUSDT"}  # tablename and symbol


def interval_to_milliseconds(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60 * 1000
    elif interval.endswith("h"):
        return int(interval[:-1]) * 60 * 60 * 1000
    elif interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Invalid interval format: {interval}")


def get_binance_data(
    symbol: str,
    start_time: int,
    end_time: Optional[int] = None,
    interval: str = "1m",
    limit: int = 1000,
) -> pd.DataFrame:
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    url = base_url + endpoint
    # Convert interval to milliseconds
    interval_in_milliseconds = interval_to_milliseconds(interval)

    # If no end_time is given, default to the current time
    # if end_time is None:
    #     end_time = int(datetime.datetime.now().timestamp() * 1000)
    time_range = end_time - start_time  # total time range
    pprint(f"time_range: {time_range}")
    request_max = limit * interval_in_milliseconds
    pprint(f"request_max: {request_max}")

    start_iteration = start_time
    end_iteration = start_time + request_max

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
    }

    if end_time is not None:
        params["endTime"] = end_time

    response_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    if time_range <= request_max:  # time range selected within 1000 rows limit
        resp = requests.get(url=url, params=params)
        data = resp.json()
        df = pd.DataFrame(data, columns=response_columns)

        time.sleep(1)

    elif (
        time_range > request_max
    ):  # start_time and end_time selected > limit rows of data
        df = pd.DataFrame()  # empty dataframe to append to
        num_iterations = math.ceil(time_range / request_max)  # number of loops required
        pprint(f"num_iterations: {num_iterations}")

        for i in range(num_iterations):
            # make request with updated params
            resp = requests.get(url=url, params=params)
            data = resp.json()
            _df = pd.DataFrame(data, columns=response_columns)

            df = pd.concat([df, _df])

            start_iteration = end_iteration
            end_iteration = min(
                end_iteration + request_max, end_time
            )  # don't go beyond the actual end time
            # adjust params

            params["startTime"], params["endTime"] = (
                start_iteration,
                end_iteration,
            )  # adjust params
            time.sleep(1)

    df.insert(0, "utc_datetime", pd.to_datetime(df["open_time"], unit="ms"))
    return df


def generate_bq_schema_from_pandas(df: pd.DataFrame):
    """
    Convert pandas dtypes to BigQuery dtypes.

    Parameters
    ----------
    dtypes : pandas Series
        The pandas dtypes to convert.

    Returns
    -------
    List[google.cloud.bigquery.SchemaField]
        The corresponding BigQuery dtypes.
    """
    dtype_mapping = {
        "int64": bigquery.enums.SqlTypeNames.INT64,
        "float64": bigquery.enums.SqlTypeNames.FLOAT64,
        "object": bigquery.enums.SqlTypeNames.STRING,
        "bool": bigquery.enums.SqlTypeNames.BOOL,
        "datetime64[ns]": bigquery.enums.SqlTypeNames.DATETIME,
    }

    schema = []

    for column, dtype in df.dtypes.items():
        if str(dtype) not in dtype_mapping:
            raise ValueError(f"Cannot convert {dtype} to a BigQuery data type.")

        bq_dtype = dtype_mapping[str(dtype)]
        field = bigquery.SchemaField(name=column, field_type=bq_dtype, mode="NULLABLE")
        schema.append(field)

    return schema


def update_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df["time_updated"] = datetime.datetime.now()
    df["source"] = "binance"
    df["source_type"] = "spot"
    return df


def upload_latest_data(
    symbol: str,
    interval: str,
    project_id: str,
    google_application_credentials: str,
    bucket_name: str = None,
    table_name: str = None,  # for example bigquery table id
    dataset: str = None,  # for example bigquery dataset
    start_time: int = None,
):
    gcs = GCS(
        project_id=project_id,
        google_application_credentials=google_application_credentials,
        bucket_name=bucket_name,
    )

    bq = BigQuery(
        project_id=project_id,
        google_application_credentials=google_application_credentials,
        dataset=dataset,
        table_name=table_name,
    )

    # flag to check if dataset exists
    dataset_exists = bq.check_if_dataset_exists()

    # flag to check if table exists
    table_exists = bq.check_if_table_exists()

    # if dataset or table does not exist, create them
    if not dataset_exists or not table_exists:
        logger.warning("Dataset or table does not exist. Creating them now...")
        assert (
            start_time is not None
        ), "start_time must be provided to create dataset and table"

        time_now = int(datetime.datetime.now().timestamp() * 1000)

        df = get_binance_data(
            symbol=symbol,
            start_time=start_time,
            end_time=time_now,
            interval=interval,
            limit=1000,
        )
        df = update_metadata(df)
        pprint(df)
        schema = generate_bq_schema_from_pandas(df)
        pprint(schema)

        bq.create_dataset()
        bq.create_table(schema=schema)  # empty table with schema
        job_config = bq.load_job_config(schema=schema, write_disposition="WRITE_APPEND")
        bq.load_table_from_dataframe(df=df, job_config=job_config)
    else:
        logger.info("Dataset and table already exist. Fetching the latest date now...")

        # Query to find the maximum open_date
        query = f"""
        SELECT MAX(open_time) as max_open_time
        FROM `{bq.table_id}`
        """
        max_date_result: pd.DataFrame = bq.query(query, as_dataframe=True)
        pprint(max_date_result)
        max_open_time = max(max_date_result["max_open_time"])
        pprint(max_open_time)

        # now max_open_time is your new start_time
        start_time = max_open_time + interval_to_milliseconds(interval)
        time_now = int(datetime.datetime.now().timestamp() * 1000)

        # only pull data from start_time onwards, which is the latest date in the table
        df = get_binance_data(
            symbol="BTCUSDT",
            start_time=start_time,
            end_time=time_now,
            interval="1m",
            limit=1000,
        )
        df = update_metadata(df)
        # Append the new data to the existing table
        job_config = bq.load_job_config(write_disposition="WRITE_APPEND")
        bq.load_table_from_dataframe(df=df, job_config=job_config)

    # # Save the data to a local CSV file
    # final_data.to_csv(f"{table_name}.csv", index=False)

    # # Create a GCPConnector instance
    # gcp = GCPConnector(project_id, service_account_key_json, bucket_name)

    # # Upload the file to GCS
    # blob_name = f"staging/{table_name}.csv"
    # gcp.upload_blob(f"{table_name}.csv", blob_name)

    # # Create a BigQuery instance
    # bq = BigQuery(project_id, service_account_key_json)

    # # Define the schema of your BigQuery table
    # # TODO: Adjust this according to your actual schema
    # schema = [
    #     bigquery.SchemaField("open_time", "TIMESTAMP"),
    #     bigquery.SchemaField("open", "FLOAT64"),
    #     bigquery.SchemaField("high", "FLOAT64"),
    #     # Add the rest of your columns here...
    # ]

    # # Load the data from GCS to BigQuery
    # gcs_uri = f"gs://{bucket_name}/{blob_name}"
    # bq.load_gcs_to_bq(gcs_uri, "staging", table_name, schema)

    # print(f"{str(len(final_data))} added to table {table_name}")


# eg: int(datetime.datetime(2023, 6, 1, 8, 0, 0).timestamp() * 1000)
start_time = int(datetime.datetime(2023, 6, 1, 20, 0, 0).timestamp() * 1000)

upload_latest_data(
    "BTCUSDT",  # "ETHUSDT
    "1m",
    PROJECT_ID,
    GOOGLE_APPLICATION_CREDENTIALS,
    BUCKET_NAME,
    dataset="mlops_pipeline_v1_staging",
    table_name="binance_btcusdt_spot",
    start_time=start_time,
)
