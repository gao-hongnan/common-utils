"""This module aims to mimic the functionality of Great Expectations, but with a
very minimalistic approach just to illustrate the concept.

TODO: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#data_and_model_validation
    - Add more validation methods.
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict

import pandas as pd
from tabulate import tabulate

from common_utils.core.logger import Logger

# Setup logging
logger = Logger(
    module_name=__name__, propagate=False, log_root_dir=None, log_file=None
).logger


class DataFrameValidator:
    """Validate pandas DataFrame against specified schema.

    Attributes
    ----------
    df : pd.DataFrame
        The dataframe to be validated.
    schema : Dict[str, Any]
        The expected schema of the dataframe as a dictionary.
        The dictionary keys should be column names and the values should be the expected dtypes.
    """

    def __init__(self, df: pd.DataFrame, schema: Dict[str, Any]) -> None:
        self.df = df
        self.schema = schema

    def validate_missing(self) -> DataFrameValidator:
        """Check if there are missing values in the dataframe."""
        logger.info("Checking missing values...")
        # note missing is the number of missing values per column
        missing: pd.Series = self.df.isnull().sum().sort_values(ascending=False)

        percent: pd.Series = (
            self.df.isnull().sum() / self.df.isnull().count()
        ).sort_values(ascending=False)

        missing_data = pd.concat(
            [missing, percent], axis=1, keys=["Missing", "Percent"]
        )
        missing_data.index.name = "Feature"

        if missing_data["Missing"].any():
            logger.warning(  # pylint: disable=logging-not-lazy
                "The following columns have missing data:\n"
                + textwrap.indent(
                    tabulate(
                        missing_data[missing_data["Missing"] > 0],
                        headers="keys",
                        tablefmt="psql",
                    ),
                    "    ",
                )
            )
        return self

    def validate_data_types(self) -> DataFrameValidator:
        """Check if the data types of the dataframe's columns match the expected schema."""
        logger.info("Checking data types...")
        for column, dtype in self.schema.items():
            if self.df[column].dtype != dtype:
                logger.warning(
                    f"The data type for {column} is not {dtype}, it's {self.df[column].dtype}"
                )
        return self

    def validate_schema(self) -> DataFrameValidator:
        """Check if the schema of the dataframe matches the expected schema."""
        logger.info("Checking schema...")
        for column in self.schema.keys():
            if column not in self.df.columns:
                logger.warning(
                    f"Expected {column} in the dataframe, but it's not present."
                )
        for column in self.df.columns:
            if column not in self.schema.keys():
                logger.warning(f"Found unexpected column {column} in the dataframe.")
        return self

    def validate_all(self) -> None:
        """Perform all validations on the dataframe."""
        self.validate_schema().validate_data_types().validate_missing()
