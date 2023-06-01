FROM python:3.9-slim

RUN pip install prefect

EXPOSE 4200

CMD prefect agent start \
    --work-queue ${PREFECT__CLOUD__AGENT__WORKER_QUEUE}
