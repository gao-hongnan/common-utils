# MLFlow Remote Tracking Server

## Method 1. GCP VM

```bash
gcloud compute ssh --zone "asia-southeast1-a" "mlops-pipeline-v1" --project "gao-hongnan"
```

```bash
gaohn@<VM_NAME> $ git clone https://github.com/gao-hongnan/common-utils.git
```

```bash
gaohn@<VM_NAME> $ cd common-utils/examples/containerization/docker/mlflow
```

Then we echo something like the below to `.env` file.

```bash
echo -e "# Workspace storage for running jobs (logs, etc)\n\
WORKSPACE_ROOT=/tmp/workspace\n\
WORKSPACE_DOCKER_MOUNT=mlflow_workspace\n\
DB_DOCKER_MOUNT=mlflow_db\n\
\n\
# db\n\
POSTGRES_VERSION=13\n\
POSTGRES_DB=mlflow\n\
POSTGRES_USER=postgres\n\
POSTGRES_PASSWORD=mlflow\n\
POSTGRES_PORT=5432\n\
\n\
# mlflow\n\
MLFLOW_IMAGE=mlflow-docker-example\n\
MLFLOW_TAG=latest\n\
MLFLOW_PORT=5001\n\
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:password@db:5432/mlflow\n\
MLFLOW_ARTIFACT_STORE_URI=gs://gaohn/imdb/artifacts" > .env
```

Finally, run `bash build.sh` to build the docker image and run the container.

Once successful, you can then access the MLFlow UI at `http://<EXTERNAL_VM_IP>:5001`.
