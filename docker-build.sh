#!/bin/bash

echo "Building and starting Docker environment..."

mkdir -p infrastructure/prometheus
mkdir -p infrastructure/grafana/provisioning/datasources
mkdir -p infrastructure/grafana/dashboards

cp infrastructure/prometheus/prometheus.yml infrastructure/prometheus/prometheus.yml

docker-compose build 
docker-compose up -d

echo "Environment started! Services available at:"
echo "- API: http://localhost:8080"
echo "- API Docs: http://localhost:8080/docs"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"