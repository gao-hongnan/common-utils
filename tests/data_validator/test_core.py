from typing import Any, Dict, Tuple

import pandas as pd
import pytest

from common_utils.data_validator.core import DataFrameValidator


@pytest.fixture(
    params=[
        (
            pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]}),
            {"A": "int64", "B": "object", "C": "float64"},
        )
    ]
)
def df_schema_pair(request) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pytest fixture to provide dataframe and schema for testing.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The request object for fixture requests.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        The dataframe and schema to be tested.
    """
    return request.param


def test_DataFrameValidator(
    df_schema_pair: Tuple[pd.DataFrame, Dict[str, Any]]
) -> None:
    """Test the DataFrameValidator class.

    Parameters
    ----------
    df_schema_pair : Tuple[pd.DataFrame, Dict[str, Any]]
        The dataframe and schema to be tested.
    """
    df, schema = df_schema_pair
    df_validator = DataFrameValidator(df, schema)
    df_validator.validate()
