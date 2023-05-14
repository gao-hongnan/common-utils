check_data_file() {
    if [[ -z "${DATA_FILE}" ]]; then
        echo "Error: Data file not provided"
        usage
    fi
}
track_data_file_with_dvc() {
    dvc add "${DATA_FILE}"
    DATA_DVC_FILE="${DATA_FILE}.dvc"
    git add "${DATA_DVC_FILE}"
    git commit -m "Track data file with DVC"
}

push_data_to_remote() {
    if [[ -n "${REMOTE_URL}" && -n "${BUCKET}" && -n "${PATH}" ]]; then
        dvc push
    fi
}
track_data_file_with_dvc
push_data_to_remote