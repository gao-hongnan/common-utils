FROM python:3.9-slim

RUN pip install protobuf mlflow==2.3.1 psycopg2-binary

CMD mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_STORE_URI} \
    --host 0.0.0.0

