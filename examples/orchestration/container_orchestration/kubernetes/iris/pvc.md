Here's a step by step guide to set up a PersistentVolume (PV) and a PersistentVolumeClaim (PVC) in a local Kubernetes environment.

1. First, you need to create a PersistentVolume. Below is an example of a PV that uses a `hostPath` volume plugin. You would replace `/Users/reighns/Downloads/iris/model` with the directory containing your model weights.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: "/Users/reighns/Downloads/iris/model"
```

Save this to a file (e.g., `pv.yaml`), and then apply it with `kubectl apply -f pv.yaml`.

2. Now you can create a PersistentVolumeClaim that will bind to the PersistentVolume you just created.

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

Save this to a file (e.g., `pvc.yaml`), and then apply it with `kubectl apply -f pvc.yaml`.

3. Now, you should be able to use this PVC in your Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-server-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi-server-deployment
  template:
    metadata:
      labels:
        app: fastapi-server-deployment
    spec:
      containers:
      - name: fastapi-server-deployment
        image: ttl.sh/john_doe_iris/fastapi-server-image:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: model-volume
          mountPath: /model
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

Apply this Deployment with `kubectl apply -f deployment.yaml`.

Now, your application should be able to read the model weights from the `/model` directory inside the container, which is backed by the `model-pvc` PVC, which is bound to the `model-pv` PV, which is backed by the directory on your local filesystem (`/Users/reighns/Downloads/iris/model`).