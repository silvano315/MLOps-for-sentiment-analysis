# Dockerfile for the main application

# Official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and setup
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install needed packages in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Copy current directory contents
COPY app/ ./app/
COPY config/ ./config/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Make ports available to the world outside the container
EXPOSE 8080 8000

# Define Environment variables
ENV PYTHONPATH=/app
ENV MODEL_CACHE_DIR=/app/models/cache
ENV HF_HOME=/app/models/cache

# Directory creation for static, cache and volume
RUN mkdir -p /app/static
RUN mkdir -p /app/models/cache && chmod -R 777 /app/models/cache
RUN mkdir -p /app/data/datasets

# Initialize Container to download and preprocess datasets
COPY scripts/init_container.sh /app/scripts/init_container.sh
RUN chmod +x /app/scripts/init_container.sh

# User for security
RUN useradd -m appuser
USER appuser

# RUN app.py when the container launches
ENTRYPOINT [ "/app/scripts/init_container.sh" ]
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]