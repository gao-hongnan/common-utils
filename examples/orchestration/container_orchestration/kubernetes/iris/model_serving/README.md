# Model Serving

SEE https://github.com/aisingapore/ml-project-cookiecutter-gcp/tree/master/%7B%7Bcookiecutter.repo_name%7D%7D/aisg-context/k8s/model-serving-api

Here's how the FastAPI application could look:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
model = joblib.load("/model/iris_model.pkl")

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(iris: Iris):
    data = iris.dict()
    df = pd.DataFrame(np.array(list(data.values())).reshape(-1, len(data)))
    prediction = model.predict(df)
    return {'prediction': prediction.tolist()[0]}
```

In this FastAPI app, we use a Pydantic model `Iris` to define the data structure for the incoming data. We then use the `/predict` endpoint to accept POST requests, where the data payload is expected to be of the `Iris` type. The data is then used for model prediction.

You can still use the same Dockerfile as the previous example, but you need to replace `flask` with `fastapi` and `uvicorn` in the requirements.txt, and change the CMD line in Dockerfile to run the FastAPI application:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory in the container
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note the change in the `CMD` command in the Dockerfile. FastAPI uses Uvicorn as the ASGI server to run the application, and by default, FastAPI applications run on port 8000.

Now to build the Docker image:

```bash
docker build -t model-server-image:latest .
docker build -t ttl.sh/john_doe_iris/model-server-image:latest . # for ttl.sh
```

and to push:

```bash
docker push ttl.sh/john_doe_iris/model-server-image:latest
```


Now, we can deploy this model server to Kubernetes. We will use a Deployment and a Service to deploy the model server. The Deployment will ensure that there are always 3 replicas of the model server running, and the Service will expose the model server to the outside world.

```yaml
# model-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: model-server-image:latest
        ports:
        - containerPort: 8000
```


# model-server-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server-service
spec:
  selector:
    app: model-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```
