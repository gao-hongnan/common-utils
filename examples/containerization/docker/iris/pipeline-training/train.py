import logging
import os
import random
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier  # pylint: disable=unused-import
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def seed_all(seed: int = 42) -> None:
    """Seed all random number generators."""
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)  # numpy pseudo-random generator
    random.seed(seed)  # python's built-in pseudo-random generator


class Config(SimpleNamespace):
    """Configuration for the app."""

    def __init__(  # pylint: disable=useless-super-delegation, useless-parent-delegation
        self, **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(**kwargs)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Metadata:
    """Metadata to track internal state."""

    data_dir: Path = None
    artifacts_dir: Path = None
    filepath: Path = None

    X: pd.DataFrame = None
    y: pd.Series = None
    le: LabelEncoder = None


def download_iris_data(config: Config, metadata: Metadata) -> Metadata:
    """
    This function downloads the iris dataset from the provided URL.
    """
    # with timestamp so each run is unique
    data_dir: Path = Path(config.stores_dir) / config.timestamp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath: Path = data_dir / config.filename
    urllib.request.urlretrieve(config.url, filepath)

    metadata.data_dir = data_dir
    metadata.filepath = filepath
    return metadata


def load_and_preprocess_data(config: Config, metadata: Metadata) -> Metadata:
    """
    This function loads and preprocesses the data.
    It returns the features X, labels y and the label encoder.
    """

    # Load the data with the column names
    df = pd.read_csv(metadata.filepath, names=config.column_names)

    # Prepare data
    le = LabelEncoder()
    df[config.target_name] = le.fit_transform(df[config.target_name])

    # Split the data into features and labels
    X = df.drop(config.target_name, axis=1)
    y = df[config.target_name]

    metadata.X = X
    metadata.y = y
    metadata.le = le
    return metadata


def train_model(config: Config, metadata: Metadata) -> Metadata:
    """
    This function trains a random forest classifier and returns the trained model.
    """
    model_name: str = config.model_config.pop("name")
    model: BaseEstimator = globals()[model_name](**config.model_config)
    model.fit(metadata.X, metadata.y)

    # Evaluate the model
    accuracy = accuracy_score(metadata.y, model.predict(metadata.X))
    LOGGER.info(f"Accuracy: {accuracy}")

    # Save the trained model weights
    artifacts_dir: Path = Path(config.stores_dir) / config.timestamp / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "model.joblib")

    joblib.dump(metadata.le, artifacts_dir / "label_encoder.joblib")

    return metadata


def run(config: Config, metadata: Metadata) -> Metadata:
    """Run the training pipeline."""
    seed_all(config.seed)

    metadata = download_iris_data(config, metadata)

    # Load and preprocess the data
    metadata = load_and_preprocess_data(config, metadata)

    # Train the model
    metadata = train_model(config, metadata)
    return metadata


if __name__ == "__main__":
    config = Config(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        stores_dir="./stores",
        filename="iris.csv",
        column_names=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ],
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        target_name="species",
        seed=42,
        model_config={
            "name": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42,
        },
    )
    metadata = Metadata()
    metadata = run(config, metadata)
