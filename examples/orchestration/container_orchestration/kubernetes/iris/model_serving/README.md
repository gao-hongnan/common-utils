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
docker build -t fastapi-server-image:latest .
docker build -t ttl.sh/john_doe_iris/fastapi-server-image:latest . # for ttl.sh
```

and to push:

```bash
docker push ttl.sh/john_doe_iris/fastapi-server-image:latest
```


Now, we can deploy this model server to Kubernetes. We will use a Deployment and a Service to deploy the model server. The Deployment will ensure that there are always 3 replicas of the model server running, and the Service will expose the model server to the outside world.

```yaml
# fastapi-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi-server
  template:
    metadata:
      labels:
        app: fastapi-server
    spec:
      containers:
      - name: fastapi-server
        image: fastapi-server-image:latest
        ports:
        - containerPort: 8000
```

```bash
kubectl apply -f fastapi-deployment.yaml
```


```yaml
# fastapi-server-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-server-service
spec:
  selector:
    app: fastapi-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```

```bash
kubectl apply -f fastapi-service.yaml
```

## Setup PVC

The error message you're seeing indicates that the PersistentVolumeClaim (PVC) "model-pvc" cannot be found. This means that your Deployment is trying to mount a volume using a PVC that doesn't exist in your cluster.

In Kubernetes, a PVC is a request for storage that can be fulfilled by a PersistentVolume (PV). In this case, you've set up your Deployment to use a PVC named "model-pvc", but Kubernetes can't find a PVC with that name.

To fix this, you should create a PVC named "model-pvc". Here's an example of how you might define this PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

This PVC requests a PV with a capacity of 1Gi and an access mode of `ReadWriteOnce`, which means the volume can be mounted as read-write by a single node.

You can create the PVC in your cluster using the `kubectl apply` command:

```bash
kubectl apply -f model-pvc.yaml
```

Once the PVC is created, Kubernetes will try to bind it to a PV that satisfies the PVC's requirements. If such a PV exists, the PVC is bound to the PV. If no such PV exists, and dynamic provisioning has been set up, a PV is provisioned according to the StorageClass of the PVC.

Once the PVC is bound to a PV, your Deployment should be able to mount the volume and the pods should start successfully. If the pods are still not starting, you can check the status of the PVC to make sure it's bound to a PV:

```bash
kubectl get pvc model-pvc
```

This will show you the status of the PVC. If the PVC is bound to a PV, the status will be `Bound`. If the PVC is not bound, the status will be `Pending`, and you'll need to investigate why a suitable PV isn't available.

## PVC and PV

you can use PersistentVolumeClaims (PVCs) in your local Kubernetes development environment, such as Minikube or Docker Desktop. PVCs are a way for your applications to request specific sizes and access modes for PersistentVolumes (PVs), such as read/write once or read/write many.

However, it's worth mentioning that for PVCs to work, there needs to be a corresponding PV available that satisfies the claim's requirements, or a dynamic provisioner should be set up that can create such a volume on-demand. The specific configuration may depend on your development environment.

Here's an example of how you can manually create a PV and PVC on your local machine:

First, create a PV. This will just reserve storage for use. You will also define the access modes and capacity:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv
spec:
  capacity:
    storage: 1Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  local:
    path: /path/to/your/local/directory
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - minikube  # or the name of your local node
```

Then, create a PVC. The PVC will look for a PV that satisfies its requirements and bind to it:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

Please replace `/path/to/your/local/directory` with the directory where your model file is stored on your local machine, and `minikube` with the name of your local Kubernetes node (which may be "minikube" if you are using Minikube, or "docker-desktop" if you are using Docker Desktop).

Remember to apply these with `kubectl apply -f pv.yaml -f pvc.yaml`.

Once you have these set up, you can use the PVC in your deployment as you described in your previous message. Kubernetes will mount the PV associated with the PVC to the specified path in the container.

The concept of PersistentVolumes (PVs) and PersistentVolumeClaims (PVCs) is crucial when working with Kubernetes.

**PersistentVolumes (PVs)**: Kubernetes provides a storage abstraction system to handle the complexity of different storage options and vendors. PV is a part of this abstraction system. A PV is a piece of storage in the cluster that has been manually provisioned by an administrator or dynamically provisioned by Kubernetes using Storage Classes. It's a resource in the cluster just like a node is a cluster resource. PVs are volume plugins like Volumes but have a lifecycle independent of any individual Pod that uses the PV. This object describes the storage capacity, how that volume is accessed, and what to do with the volume when it's no longer needed.

**PersistentVolumeClaims (PVCs)**: While PVs are resources available for use in a cluster, PVCs are requests for those resources. They are a way for users to "claim" durable storage (such as a GCE Persistent Disk or an iSCSI volume) without knowing the details of the particular cloud environment. PVCs can request specific sizes, access modes (e.g., they can be mounted once read/write or many times read-only), or more generally, characteristics described through Storage Classes.

The important takeaway is that PVs abstract the underlying storage systems (whether that's cloud storage, network storage, or locally attached storage), and PVCs let applications and services request particular storage resources without needing to know about the underlying technology. This abstraction makes it easier to work with storage in Kubernetes and allows your applications to be portable across different environments.

In practical terms, this means you could develop an application that uses a PVC for storage in your local development environment (with a local directory serving as the PV), and then deploy the same application in a cloud environment (with a cloud storage bucket as the PV), with no changes to the application configuration or code. The application only knows that it has a claim on a certain amount of storage - it doesn't need to know anything about where or how that storage is provided.