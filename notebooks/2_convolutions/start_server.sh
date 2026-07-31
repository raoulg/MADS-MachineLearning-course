#!/bin/bash

# Check if mlflow is installed
if ! command -v mlflow &> /dev/null
then
    echo "mlflow could not be found, please install mlflow and activate venv."
    exit
fi

# The store lives at the root of the repository (planning/conventions.md section 5), so runs from
# every lesson land in one place and there is one file to delete for a clean slate.
STORE="$(cd "$(dirname "$0")/../.." && pwd)/mlruns.db"

# Run the mlflow server
mlflow server \
    --backend-store-uri "sqlite:///${STORE}" \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1
