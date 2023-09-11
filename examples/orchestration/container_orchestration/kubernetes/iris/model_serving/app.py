import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("/model/iris_model.pkl")


class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(iris: Iris):
    data = iris.model_dump(mode="python")
    df = pd.DataFrame(np.array(list(data.values())).reshape(-1, len(data)))
    prediction = model.predict(df)
    proba = model.predict_proba(df).tolist()[0]  # Prediction probabilities

    # Log data and predictions for monitoring
    log_entry = {"input": data, "prediction": prediction.tolist()[0], "proba": proba}
    with open("/model/predictions_log.csv", "a") as f:
        f.write(f"{log_entry}\n")

    return {"prediction": prediction.tolist()[0], "proba": proba}
