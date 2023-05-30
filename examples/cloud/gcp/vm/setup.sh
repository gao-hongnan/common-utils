git clone https://github.com/gao-hongnan/common-utils.git
cd common-utils/scripts/containerization/docker
bash docker_setup.sh

# setup mlflow
gcloud compute firewall-rules create allow-mlflow --allow tcp:8001 --source-ranges 0.0.0.0/0
cd ~/common-utils/examples/containerization/docker/mlflow
bash build.sh # create .env

# setup airbyte
# https://docs.airbyte.com/deploying-airbyte/on-gcp-compute-engine/
gcloud compute firewall-rules create allow-airbyte --allow tcp:8000 --source-ranges 0.0.0.0/0