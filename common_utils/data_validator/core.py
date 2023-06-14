"""This module aims to mimic the functionality of Great Expectations, but with a
very minimalistic approach just to illustrate the concept."""
from __future__ import annotations
from typing import Any, Dict

import pandas as pd
import pytest

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

    def check_missing(self) -> DataFrameValidator:
        """Check if there are missing values in the dataframe."""
        missing_data = self.df.isnull().sum()
        if missing_data.any():
            logger.warning("The following columns have missing data:")
            logger.warning(missing_data[missing_data > 0])
        return self

    def check_data_types(self) -> DataFrameValidator:
        """Check if the data types of the dataframe's columns match the expected schema."""
        for column, dtype in self.schema.items():
            if self.df[column].dtype != dtype:
                logger.warning(
                    f"The data type for {column} is not {dtype}, it's {self.df[column].dtype}"
                )
        return self

    def check_schema(self) -> DataFrameValidator:
        """Check if the schema of the dataframe matches the expected schema."""
        for column in self.schema.keys():
            if column not in self.df.columns:
                logger.warning(
                    f"Expected {column} in the dataframe, but it's not present."
                )
        for column in self.df.columns:
            if column not in self.schema.keys():
                logger.warning(f"Found unexpected column {column} in the dataframe.")
        return self

    def validate(self) -> None:
        """Perform all validations on the dataframe."""
        self.check_schema().check_data_types().check_missing()
