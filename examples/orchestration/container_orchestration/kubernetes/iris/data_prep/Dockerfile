# Dockerfile

FROM python:3.8-slim

WORKDIR /app

COPY data_prep.py ./

RUN pip install pandas

CMD ["python", "data_prep.py"]
