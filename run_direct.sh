#!/bin/bash

source mlflow-env/bin/activate
source secrets.sh

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:///tmp/mlruns}
export BACKEND_STORE_URI=${BACKEND_STORE_URI:-file:///tmp/mlruns}
export ARTIFACT_ROOT=${ARTIFACT_ROOT:-s3://ppml2026/mlflow-artifacts}

echo "============================================"
echo "MLflow Iris Training - Image Docker Hub"
echo "============================================"

echo "Using image: pkodocker/iris-mlflow:latest"

rm -rf /tmp/mlruns/*

docker run --rm \
  -v "$(pwd):/app" \
  -e MLFLOW_TRACKING_URI \
  -e BACKEND_STORE_URI \
  -e ARTIFACT_ROOT \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  pkodocker/iris-mlflow:latest
