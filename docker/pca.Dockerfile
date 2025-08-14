FROM python:3.11-slim

WORKDIR /app

COPY multiverse ./multiverse
COPY docker/requirements-pca.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.models.pca", "--config_path", "./config_alldatasets.json"]
