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
bash common-utils/scripts/dataops/airbyte/airbyte_setup.sh -u <USERNAME> -p <PASSWORD>
# package octavia here so you can version TODO

# expose ports
gcloud compute firewall-rules create energy-forecasting-expose-ports \
    --allow tcp:8501,tcp:8502,tcp:8001,tcp:8000,tcp:8080,tcp:5000 \
    --target-tags=energy-forecasting-expose-ports \
    --description="Firewall rule to expose ports for energy forecasting" \
    --project=gao-hongnan

#  IAP (Identity-Aware Proxy) TCP tunneling.
gcloud compute firewall-rules create iap-tcp-tunneling \
    --allow tcp:22 \
    --target-service-accounts=gcp-storage-service-account@gao-hongnan.iam.gserviceaccount.com \
    --source-ranges=35.235.240.0/20 \
    --description="Firewall rule to allow IAP TCP tunneling" \
    --project=gao-hongnan

vm_create --instance-name gaohn-energy-forecast \
    --machine-type e2-standard-2 \
    --zone us-west2-a \
    --boot-disk-size 20GB \
    --image ubuntu-1804-bionic-v20230510 \
    --image-project ubuntu-os-cloud \
    --project gao-hongnan \
    --service-account gcp-storage-service-account@gao-hongnan.iam.gserviceaccount.com \
    --scopes https://www.googleapis.com/auth/cloud-platform \
    --description "Energy Consumption VM instance" \
    --additional-flags --tags=http-server,https-server


gcloud compute instances create ml-pipeline \
    --project=gao-hongnan \
    --zone=us-west2-a \
    --machine-type=e2-standard-2 \
    --boot-disk-size=20GB \
    --image-family=ubuntu-1804-lts \
    --image-project=ubuntu-os-cloud \
    --service-account=gcp-storage-service-account@gao-hongnan.iam.gserviceaccount.com  \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server,https-server,energy-forecasting-expose-ports \
    --description="Energy Consumption VM instance"
