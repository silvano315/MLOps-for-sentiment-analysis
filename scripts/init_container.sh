#!/bin/bash

echo "Initializing container..."

# Environment
export PYTHONPATH=/app

# Script for checking dataset availabilty to be downloaded
if [ ! -f "/app/data/datasets/cached/.initialized" ]; then
    echo "Downloading datasets..."
    python -m data.datasets.download_datasets --save-dir /app/data/datasets/cached
    touch /app/data/datasets/cached/.initialized
    echo "Datasets downloaded successfully!"
else
    echo "Datasets already downloaded"
fi

echo "Starting application..."
exec "$@"