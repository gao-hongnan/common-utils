FROM python:3.9-slim

RUN pip install prefect

EXPOSE 4200

RUN echo ${PREFECT__CLOUD__AUTH_TOKEN}
RUN echo ${PREFECT__CLOUD__WORKSPACE}
# set prefect cloud API key
RUN prefect cloud login --key ${PREFECT__CLOUD__AUTH_TOKEN} --workspace ${PREFECT__CLOUD__WORKSPACE}

CMD prefect agent start \
    --work-queue ${PREFECT__CLOUD__AGENT__WORKER_QUEUE}
