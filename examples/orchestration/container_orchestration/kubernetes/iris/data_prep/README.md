# Data Prep

In this scenario, let's assume you're working with a public dataset that can be
easily downloaded and ingested for use in model training.

Please note that this example is greatly simplified. In a real-world scenario,
you would need to account for various additional factors such as security, fault
tolerance, and efficiency.

For data ingestion, we might have a simple Python script that downloads a
dataset from a public URL, cleans it, and stores it in a CSV file.

Let's start with a simple Python script for data preparation, which we will run
in a Kubernetes Pod. The script will download the Iris dataset from a public
URL, and write it to a CSV file:

```python
# data_prep.py

import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
data = pd.read_csv(url, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
data.to_csv('/data/iris.csv', index=False)
```

This script downloads the dataset and saves it as a CSV file.

Now let's create a Dockerfile that builds an image with this script:

```Dockerfile
# Dockerfile

FROM python:3.8-slim

WORKDIR /app

COPY data_prep.py ./

RUN pip install pandas

CMD ["python", "data_prep.py"]
```

```bash
docker build -t data-prep-image:latest .
docker build -t ttl.sh/john_doe_iris/data-prep-image:latest . # for ttl.sh
```

and to run:

```bash
docker run -v ~/Downloads/iris:/data data-prep-image:latest
docker run -v ~/Downloads/iris:/data ttl.sh/john_doe_iris/data-prep-image:latest # for ttl.sh
```

and to push:

```bash
docker push ttl.sh/john_doe_iris/data-prep-image:latest
```

You can build this Docker image and push it to a Docker registry. Assuming
you've named the image `data-prep-image`, you can create a Kubernetes Job to run
this script:

```yaml
# data-prep-job.yaml

apiVersion: batch/v1
kind: Job
metadata:
    name: data-prep-job
spec:
    template:
        spec:
            volumes:
                - name: data-volume
                  hostPath:
                      path: /Users/reighns/gaohn/Downloads/iris
            containers:
                - name: data-prep
                  image: data-prep-image:latest
                  volumeMounts:
                      - mountPath: "/data"
                        name: data-volume
            restartPolicy: Never
```

In this YAML file, we create a Job that runs a Pod with the `data-prep-image`.
The Pod has a volume mounted to a directory on your local machine where the data
will be saved.

You can create this Job with the command:

```bash
kubectl apply -f data-prep-job.yaml
```

I got an error, so do:

```bash
kubectl logs data-prep-job-xxxxx
```

and got

```bash
Error from server (BadRequest): container "data-prep" in pod "data-prep-job-rkbb7" is waiting to start: trying and failing to pull image
```

Once the Job is complete, you should find the iris.csv file in your local data
directory.

Remember, this is a very simple example, and a real-world data ingestion and
preparation process would likely be more complex, involve multiple stages, and
require additional considerations like error handling and data privacy.

## Job vs Cronjob vs Deployment vs Service

A `Job` in Kubernetes is a controller object that represents a finite task - it creates one or more pods and ensures that a specified number of them successfully terminate. When a specified number of successful completions is reached, the job is complete. Deleting a Job will clean up the pods it created.

A Job differs from other types of Kubernetes resources:

- A `Pod` represents a single instance of a running process in a cluster and can contain one or more containers. Once a Pod is terminated, it will not be restarted. In other words, while containers in a Pod can restart, the Pod itself cannot.

- A `Deployment` is a higher-level concept that manages ReplicaSets and provides declarative updates to Pods along with a lot of other features. Therefore, unlike a Job, a Deployment is intended for long-running applications.

- A `Service` in Kubernetes is an abstraction which defines a logical set of Pods and a policy by which to access them. A service can be exposed in different ways by specifying a type in the ServiceSpec: ClusterIP, NodePort, LoadBalancer, and ExternalName.

- A `CronJob` is a special kind of Job in Kubernetes. It's like the "crontab" in Unix systems, which can be scheduled to run jobs at specific times. A CronJob object in Kubernetes is used to run Jobs on a time-based schedule, as specified in .spec.schedule in the CronJob's YAML. These are noted in the Cron format.

In essence, a `CronJob` creates `Jobs` that run on a time-based schedule. The main difference between a `Job` and a `CronJob` is that Jobs run to completion once triggered (either manually or by another process), while CronJobs run at specified times or intervals.