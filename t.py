gcloud compute instances create $VM_NAME \
    --machine-type=$MACHINE_TYPE \
    --zone=$ZONE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --image=$IMAGE \
    --image-project=$IMAGE_PROJECT \
    --project=$PROJECT_ID \
    --description="$DESCRIPTION" \

gcloud compute instances create imdb \
    --machine-type=e2-medium \
    --zone=us-west2-a \
    --boot-disk-size=10GB \
    --image=ubuntu-1804-bionic-v20230510 \
    --image-project ubuntu-os-cloud \
    --project gao-hongnan \
    --service-account gcp-storage-service-account@gao-hongnan.iam.gserviceaccount.com \
    --scopes https://www.googleapis.com/auth/cloud-platform \
    --description="IMDB VM instance"

