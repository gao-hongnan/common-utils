FROM python:3.9-slim

RUN pip install prefect

EXPOSE 4200

# set prefect cloud API key
RUN prefect cloud login --key ${PREFECT__CLOUD__AUTH_TOKEN} --workspace ${PREFECT__CLOUD___WORKSPACE}

CMD prefect agent start \
    --work-queue ${PREFECT__CLOUD__AGENT__WORKER_QUEUE}
