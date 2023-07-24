# Model Training

Sure, let's proceed to the next step in the Machine Learning (ML) pipeline - Model Training. The goal of this step is to use the prepared data to train an ML model. Here, we will write a Python script for training a simple logistic regression model using scikit-learn.

**Step 1:** Write a Python script for model training. Save this as `train.py`:

```python
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('/data/iris_processed.csv')

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.2)

# Define and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save the model
os.makedirs('/data/model', exist_ok=True)
joblib.dump(model, '/data/model/iris_model.pkl')
```

**Step 2:** Write a Dockerfile for the model training script. Save this as `Dockerfile.train`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY train.py .

CMD ["python3", "train.py"]
```

**Step 3:** Build the Docker image for the model training script:

```bash
docker build -f Dockerfile -t iris-train-image:latest .
docker build -f Dockerfile -t ttl.sh/john_doe_iris/iris-train-image:latest . # for ttl.sh
```

**Step 4:** Push the Docker image to your Docker registry:

```bash
# docker tag iris-train-image:latest ttl.sh/john_doe_iris/iris-train-image:latest
docker push ttl.sh/john_doe_iris/iris-train-image:latest
```

**Step 5:** Write a Kubernetes Job for model training. Save this as `train-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-job
spec:
  template:
    spec:
      volumes:
      - name: data-volume
        hostPath:
          path: /Users/reighns/Downloads/iris
      containers:
      - name: train
        image: ttl.sh/john_doe_iris/iris-train-image:latest
        volumeMounts:
        - mountPath: "/data"
          name: data-volume
      restartPolicy: Never
```

**Step 6:** Apply the Kubernetes Job:

```bash
kubectl apply -f train-job.yaml
```

```bash
kubectl delete pods -l job-name=train-job
kubectl delete jobs train-job
```

Handy for deleting all.

Once the job is done, it will generate a model and save it as `iris_model.pkl` in your `/Users/reighns/gaohn/Downloads/iris/model` directory.

Keep in mind that the code snippets above are relatively simple and may need to be extended based on the specific requirements of your model, data and infrastructure. For example, the model training step could include parameter tuning, feature selection, handling of imbalanced classes, etc.