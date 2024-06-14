#!/bin/bash

# Check if mlflow is installed
if ! command -v mlflow &> /dev/null
then
    echo "mlflow could not be found, please install mlflow and activate venv."
    exit
fi

# Run the mlflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1
