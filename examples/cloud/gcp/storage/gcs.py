import os

from dotenv import load_dotenv
import pandas as pd
import time
import math
import requests
from typing import List, Dict, Any
import datetime
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.cloud.gcp.storage.bigquery import BigQuery
from google.cloud.exceptions import NotFound
from rich import print
from rich.pretty import pprint
import pytz
from google.cloud import bigquery

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


def get_binance_data(
    symbol: str,
    start_time: int,
    end_time: int = 0,
    interval: str = "1m",
    limit: int = 1000,
) -> pd.DataFrame:
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    url = base_url + endpoint
    # Convert interval to milliseconds
    if interval.endswith("m"):
        interval_in_milliseconds = int(interval[:-1]) * 60 * 1000
    elif interval.endswith("h"):
        interval_in_milliseconds = int(interval[:-1]) * 60 * 60 * 1000
    elif interval.endswith("d"):
        interval_in_milliseconds = int(interval[:-1]) * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Invalid interval format: {interval}")

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
        "endTime": end_time,
    }

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
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            # "endTime": end_time,
        }
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

    df["utc_datetime"] = pd.to_datetime(df["open_time"], unit="ms")

    df.set_index("utc_datetime", inplace=True)

    return df


def generate_bq_schema_from_pandas(df, dtypes):
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


def create_dataset_if_not_exists(bq_client, dataset_id):
    from google.cloud.exceptions import NotFound

    dataset_ref = bq_client.dataset(dataset_id)

    try:
        bq_client.get_dataset(dataset_ref)  # Make an API request.
        print(f"Dataset {dataset_id} already exists.")
    except NotFound:
        print(f"Dataset {dataset_id} not found. Creating dataset...")
        bq_client.create_dataset(dataset_ref)
        print(f"Dataset {dataset_id} created.")


def upload_latest_data(
    project_id: str,
    google_application_credentials: str,
    bucket_name: str = None,
    table_name: str = None,  # for example bigquery table id
    dataset_id: str = None,  # for example bigquery dataset id
):
    # eg: int(datetime.datetime(2023, 6, 1, 8, 0, 0).timestamp() * 1000)
    start_time = int(datetime.datetime(2023, 6, 1, 20, 0, 0).timestamp() * 1000)
    time_now = int(datetime.datetime.now().timestamp() * 1000)

    df = get_binance_data(
        symbol="BTCUSDT",
        start_time=start_time,
        end_time=time_now,
        interval="1m",
        limit=1000,
    )
    pprint(df)

    bq = BigQuery(project_id, google_application_credentials)
    bq_client = bq.bigquery_client
    create_dataset_if_not_exists(bq_client, dataset_id)

    table_id = f"{project_id}.{dataset_id}.{table_name}"
    pprint(f"table_id: {table_id}")
    schema = generate_bq_schema_from_pandas(df, df.dtypes)

    try:
        # if table exists, get latest date and pull data from that date onwards
        table = bq_client.get_table(table_id)  # Make an API request.
        print(f"Table {table_id} already exists.")

        ### get max date from table ###
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}`"
        query_job = bq_client.query(query)  # Make an API request.

        # Wait for the job to complete.
        max_date_result = query_job.result().to_dataframe()

        max_date = max(max_date_result["open_time"])
        pprint(f"max_date: {max_date}")
        # Convert max_date to a timestamp in milliseconds.
        # max_date = int(max_date.timestamp() * 1000)
        ### end ###

        new_data = get_binance_data(
            symbol="BTCUSDT",
            start_time=max_date,
            end_time=0,  # sentinal value to get all data
            interval="1m",
            limit=1000,
        )
        pprint(new_data)
        increment_data = new_data.loc[new_data["open_time"] > max_date]
        # Load the DataFrame to the table
        bq.load_data_from_dataframe(
            increment_data, table_id, schema=schema, mode="WRITE_APPEND"
        )

    except NotFound:
        print(f"Table {table_id} is not found.")
        print(f"Creating table {table_id}...")
        table = bq_client.create_table(table_id)  # Make an API request.

        schema = generate_bq_schema_from_pandas(df, df.dtypes)
        bq.load_data_from_dataframe(df, table_id, schema=schema, mode="WRITE_APPEND")
        return  # exit since first time creating table

    # if len(tables) > 0:  # If table exists
    #     df_db = pd.read_sql(table_name, conn_sqlalchemy.engine)  # read table
    #     max_date = max(df_db["open_time"])  # gets latest date available in your table
    #     new_data = get_binance_data(
    #         symbol_name, max_date
    #     )  # pulls latest data from binance
    #     final_data = new_data.loc[
    #         new_data["open_time"] > max_date
    #     ]  # filters out that latest row of data
    # else:  # If table does not exist
    #     new_data = get_binance_data(symbol_name, 0)  # pulls all data from binance
    #     final_data = new_data  # all data is new data

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


upload_latest_data(
    PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS, BUCKET_NAME, "btcusdt", "staging"
)
