FROM python:3.11-slim

WORKDIR /app

COPY multiverse ./multiverse
COPY docker/requirements-mowgli.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.models.mowgli", "--config_path", "./config_alldatasets.json"]
