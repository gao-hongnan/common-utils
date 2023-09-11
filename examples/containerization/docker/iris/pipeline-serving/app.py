import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Config(SimpleNamespace):
    """Configuration for the app."""

    def __init__(  # pylint: disable=useless-super-delegation, useless-parent-delegation
        self, **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(**kwargs)
        self.stores_dir = (
            os.environ.get("STORES_DIR") if os.environ.get("STORES_DIR") else "./stores"
        )


@st.cache_resource()
def latest_subdir(path: Path) -> Path:
    """Get the latest subdirectory from a given path."""
    latest_subdir = max(
        (f for f in path.iterdir() if f.is_dir()),
        default=None,
        key=lambda x: x.stat().st_mtime,
    )
    LOGGER.info(f"Latest subdir: {latest_subdir}")
    return latest_subdir


def load_model(stores_dir: str) -> BaseEstimator:
    """Load the model weights from a joblib file."""
    stores_path = Path(stores_dir)
    latest_subdir_path = latest_subdir(stores_path)
    if latest_subdir_path is None:
        raise ValueError(f"No subdirectories found in {stores_path}")
    model_filepath = latest_subdir_path / "artifacts" / "model.joblib"
    model = joblib.load(model_filepath)
    return model


def load_label_encoder(stores_dir: str) -> LabelEncoder:
    """Load the label encoder from a joblib file."""
    stores_path = Path(stores_dir)
    latest_subdir_path = latest_subdir(stores_path)
    if latest_subdir_path is None:
        raise ValueError(f"No subdirectories found in {stores_path}")
    le_filepath = latest_subdir_path / "artifacts" / "label_encoder.joblib"
    le = joblib.load(le_filepath)
    return le


def get_user_input() -> pd.DataFrame:
    """
    This function gets the input features from the user and returns them as a dataframe.
    """
    # Get input features from the user
    sepal_length = st.number_input("Sepal Length", 0.0, 10.0, step=0.1)
    sepal_width = st.number_input("Sepal Width", 0.0, 10.0, step=0.1)
    petal_length = st.number_input("Petal Length", 0.0, 10.0, step=0.1)
    petal_width = st.number_input("Petal Width", 0.0, 10.0, step=0.1)

    # Convert the user input into a dataframe
    input_data = pd.DataFrame(
        {
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width],
        }
    )

    return input_data


def run_app(model: BaseEstimator, le: LabelEncoder) -> None:
    """Run the Streamlit app."""
    # Create the Streamlit app
    st.title("Iris Species Prediction")
    st.write("Enter the flower's measurements to predict the species.")

    # Get the user input
    input_data = get_user_input()

    # Make a prediction and show the result
    if st.button("Predict"):
        result = model.predict(input_data)
        species = le.inverse_transform(result)[0]
        st.success(f"The species of the flower is most likely {species}.")
        st.balloons()


if __name__ == "__main__":
    config = Config()
    model = load_model(config.stores_dir)
    le = load_label_encoder(config.stores_dir)

    # Run the app
    run_app(model, le)
