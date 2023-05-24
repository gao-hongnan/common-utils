FROM python:3.9-slim

RUN pip install protobuf mlflow==2.3.1 psycopg2-binary

CMD mlflow server \
    --backend-store-uri postgresql://postgres:password@db:5432/mlflow \
    --default-artifact-root gs://gao-hongnan/imdb \
    --host 0.0.0.0