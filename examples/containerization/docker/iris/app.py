import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def download_iris_data(url: str, filename: str, data_path: str = "./data") -> None:
    """
    This function downloads the iris dataset from the provided URL.
    """
    Path(data_path).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, Path(data_path) / filename)


def load_and_preprocess_data(filename: str, data_path: str = "./data") -> tuple:
    """
    This function loads and preprocesses the data.
    It returns the features X, labels y and the label encoder.
    """
    filepath = f"{data_path}/{filename}"
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    # Load the data with the column names
    df = pd.read_csv(filepath, names=column_names)

    # Prepare data
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])

    # Split the data into features and labels
    X = df.drop("species", axis=1)
    y = df["species"]

    return X, y, le


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    This function trains a random forest classifier and returns the trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


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


# URL of the iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
filename = "iris.csv"
data_path = "./data"

download_iris_data(url, filename, data_path=data_path)

# Load and preprocess the data
X, y, le = load_and_preprocess_data(filename, data_path=data_path)

# Train the model
model = train_model(X, y)

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
