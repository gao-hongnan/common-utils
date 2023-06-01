#!/bin/sh
prefect cloud login --key ${PREFECT__CLOUD__AUTH_TOKEN}
prefect agent start --work-queue ${PREFECT__CLOUD__AGENT__WORKER_QUEUE}
