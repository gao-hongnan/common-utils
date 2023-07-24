# data_prep.py

import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
data = pd.read_csv(
    url, names=["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
)
data.to_csv("/data/iris.csv", index=False)
