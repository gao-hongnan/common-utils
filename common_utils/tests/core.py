from typing import Any, List, Optional

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from common_utils.core.logger import Logger


def compare_test_case(
    actual: Any,
    expected: Any,
    description: str = "",
    logger: Optional[Logger] = None,
) -> None:
    try:
        if isinstance(actual, pd.DataFrame) or isinstance(actual, pd.Series):
            assert actual.equals(expected)
        elif isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected

        message = f"[green]Test passed:[/green] {description}"
        # If a logger is provided, log the message
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    except AssertionError:
        message = f"[red]Test failed:[/red] {description}\nExpected: {expected}, but got: {actual}"
        if logger is not None:
            logger.error(message)
        else:
            print(message)


def compare_test_cases(
    actual_list: List[Any],
    expected_list: List[Any],
    description_list: List[str],
    logger: Optional[Logger] = None,
) -> None:
    assert len(actual_list) == len(
        expected_list
    ), "Lengths of actual and expected are different."

    for i, (actual, expected, description) in enumerate(
        zip(actual_list, expected_list, description_list)
    ):
        compare_test_case(
            actual=actual,
            expected=expected,
            description=f"{description} - {i}",
            logger=logger,
        )


def compare_test_case_dsa(actual: Any, expected: Any, description: str = "") -> None:
    try:
        assert actual == expected
        display(
            HTML(f'<span style="color:green;">Test passed:</span>' f" {description}")
        )
    except AssertionError:
        display(
            HTML(
                f'<span style="color:red;">Test failed:</span>'
                f" {description}<br>"
                f"Expected: {expected}, but got: {actual}"
            )
        )


def compare_test_cases_dsa(
    actual_list: List[Any],
    expected_list: List[Any],
    description_list: List[str],
) -> None:
    assert len(actual_list) == len(
        expected_list
    ), "Lengths of actual and expected are different."

    for i, (actual, expected, description) in enumerate(
        zip(actual_list, expected_list, description_list)
    ):
        compare_test_case_dsa(
            actual=actual,
            expected=expected,
            description=f"{description} - {i}",
        )
