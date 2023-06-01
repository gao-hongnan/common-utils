FROM python:3.9-slim

RUN pip install prefect

EXPOSE 4200

# set prefect cloud API key
RUN prefect cloud login --key ${PREFECT_API_KEY} --workspace ${PREFECT_CLOUD_WORKSPACE}

CMD prefect agent start \
    --work-queue ${PREFECT_WORK_QUEUE}
