# The stock MLflow image has neither a Postgres driver (for the backend store) nor an
# S3 client (for the artifact store), and the tracking server needs both.
FROM ghcr.io/mlflow/mlflow:v3.16.0

RUN pip install --no-cache-dir psycopg2-binary boto3
