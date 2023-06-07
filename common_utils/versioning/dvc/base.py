import hashlib
import json
import os
from datetime import datetime

import pandas as pd

DATA_DIR = "data_versions"
CODE_DIR = "code_versions"
METADATA_FILE = "metadata.json"


def hash_file(filepath):
    """Generate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_data_version1(data, filename, bucket_name, storage):
    """Save version of data and upload to GCS."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Save data to a local file
    local_version_filename = os.path.join(DATA_DIR, f"{filename}.csv")
    data.to_csv(local_version_filename, index=False)

    # Compute a hash of the file's content
    file_content_hash = hash_file(local_version_filename)

    # Create a version ID using both the current time and the file content hash
    version_id = hashlib.md5(
        (str(datetime.now()) + file_content_hash).encode()
    ).hexdigest()

    # Rename the local file to include the version ID
    os.rename(
        local_version_filename,
        local_version_filename.replace(".csv", f"_{version_id}.csv"),
    )

    # Update local_version_filename to include the version ID
    local_version_filename = local_version_filename.replace(
        ".csv", f"_{version_id}.csv"
    )

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(f"{DATA_DIR}/{filename}_{version_id}.csv")
    blob.upload_from_filename(local_version_filename)

    return local_version_filename, version_id


def save_data_version(data, filename, bucket_name):
    """Save version of data and upload to GCS."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    version_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()

    local_version_filename = os.path.join(DATA_DIR, f"{filename}_{version_id}.csv")
    data.to_csv(local_version_filename, index=False)

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(f"{DATA_DIR}/{filename}_{version_id}.csv")
    blob.upload_from_filename(local_version_filename)

    return local_version_filename, version_id


def update_metadata(data_metadata, code_metadata):
    """Update metadata file."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    new_metadata = {
        "timestamp": str(datetime.now()),
        "data": data_metadata,
        "code": code_metadata,
    }
    existing_metadata.append(new_metadata)

    with open(METADATA_FILE, "w") as f:
        json.dump(existing_metadata, f, indent=4)

    return new_metadata


# Example usage:
data = pd.DataFrame(
    {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
    }
)

data_filename, data_version_id = save_data_version(data, "my_data")

metadata = update_metadata(
    {"filename": data_filename, "version_id": data_version_id},
)

print(metadata)
