FROM python:3.11-slim

WORKDIR /app

COPY multiverse ./multiverse
COPY docker/requirements-multivi.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.models.multivi", "--config_path", "./config_alldatasets.json"]
