#FROM apache/airflow:2.6.3
FROM apache/airflow:2.8.1-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    matplotlib==3.10.3 \
    pandas>=2.0.0 \
    requests>=2.31.0 \
    numpy>=1.24.0 \
    scipy>=1.9.0 \
    datasets>=3.5.1 \
    transformers>=4.51.0 \
    scikit-learn>=1.6.0 \
    pydantic>=2.11.0 \
    pydantic-settings>=2.9.0

COPY --chown=airflow:root app/ /opt/airflow/app/
COPY --chown=airflow:root data/ /opt/airflow/data/
COPY --chown=airflow:root scripts/ /opt/airflow/scripts/
COPY --chown=airflow:root config/ /opt/airflow/config/

ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"