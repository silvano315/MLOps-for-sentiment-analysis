FROM apache/airflow:2.6.3

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    matplotlib>=3.10.0 \
    pandas>=2.0.0 \
    requests>=2.31.0 \
    numpy>=1.24.0