# MLOPS-FOR-SENTIMENT-ANALYSIS

*Transforming Text into Insightful Sentiment Analysis*

![last commit](https://img.shields.io/badge/last%20commit-today-blue)
![python](https://img.shields.io/badge/python-96.0%25-blue)
![languages](https://img.shields.io/badge/languages-3-blue)

## Built with the tools and technologies:

![JSON](https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white)
![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![GNU Bash](https://img.shields.io/badge/GNU%20Bash-4EAA25?style=for-the-badge&logo=gnu-bash&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

---

## 🎯 Project Overview & Purpose

This project implements a **complete MLOps pipeline for sentiment analysis** that transforms social media texts into insights. The system classifies text into positive, negative, or neutral sentiment using state-of-the-art transformer models with production monitoring infrastructure.

### Project Development Phases

The project was developed through **four distinct phases**, each building upon the previous to create a comprehensive MLOps ecosystem:

**Phase 1: Foundation & Model Integration**
- Core sentiment analysis API with RoBERTa model
- Basic evaluation framework and health monitoring
- RESTful endpoints for single and batch predictions

**Phase 2: Infrastructure & Monitoring**
- Complete containerized stack with Docker Compose
- Real-time metrics collection with Prometheus
- Professional dashboards with Grafana
- Production monitoring infrastructure

**Phase 3: CI/CD & Automation**
- Automated testing pipeline with GitHub Actions
- Model fine-tuning with PEFT/LoRA techniques
- Multi-platform deployment (HuggingFace Spaces, DockerHub)
- MLflow experiment tracking integration

**Phase 4: Advanced MLOps Features**
- Apache Airflow workflow orchestration
- Automated model evaluation scheduling
- Statistical data drift detection
- Production alerting and quality assurance

---

## 🏗️ Architecture & Design Choices

The system follows **microservices architecture principles** with containerized components that can scale independently.

### Key Architectural Decisions

**MLOps-First Approach**: This project includes comprehensive monitoring, automated testing, and deployment pipelines that are often missing in academic projects.

**Event-Driven Monitoring**: The system uses Prometheus metrics collection with real-time alerting, enabling proactive issue detection rather than reactive troubleshooting.

**Infrastructure as Code**: All components are defined in Docker Compose with version-controlled configurations, ensuring reproducible deployments across environments.

**Modular Design**: Each component (API, monitoring, workflows) is independently deployable and scalable.

---

## 🔬 Implementation Deep Dive

### Model & AI Implementation

**Base Model Selection**: Chose `cardiffnlp/twitter-roberta-base-sentiment-latest` for its proven performance on social media text and robust pre-training on Twitter data.

**Fine-tuning Strategy**: Implemented **PEFT (Parameter Efficient Fine-Tuning)** with **LoRA (Low-Rank Adaptation)** for efficient model customization:
- **Efficiency**: Only ~0.1% of parameters are trainable
- **Quality**: Maintains base model performance while adapting to specific domains
- **Resource Optimization**: Reduces training time and computational requirements

**Model Registry**: Integrated HuggingFace Hub for model versioning and distribution, enabling easy model updates and rollbacks.

*For hands-on testing and model evaluation, see [MLOps_project_ProfAI.md](MLOps_project_ProfAI.md#phase-1-model-exploration-and-testing)*

### API Development with FastAPI

**Design**: Built with **API-first approach** to ensure the model is accessible and production-ready.

**Endpoint Architecture**:
- **Prediction Endpoints**: Single and batch sentiment analysis with confidence scores
- **Evaluation Endpoints**: Automated model assessment on multiple datasets
- **Admin Endpoints**: Health checks, metrics exposure, and system monitoring
- **Monitoring Integration**: Built-in Prometheus metrics collection

**Performance Optimizations**: Implemented request batching, model caching, and asynchronous processing to handle production loads efficiently.

*For API testing and usage examples, see [MLOps_project_ProfAI.md](MLOps_project_ProfAI.md#api-endpoints-overview)*

### Infrastructure & Containerization

**Multi-Service Architecture**: Designed with Docker Compose orchestrating four core services:
- **Sentiment API**: FastAPI application with model inference
- **Prometheus**: Time-series metrics collection and storage
- **Grafana**: Professional dashboards and visualization
- **PostgreSQL**: Metadata storage for workflow management

**Scalability Considerations**: Services are designed to scale horizontally with load balancing support and resource isolation.

**Production Readiness**: Includes health checks, graceful shutdowns, and persistent data volumes for production deployment.

*For infrastructure setup and scaling, see [MLOps_project_ProfAI.md](MLOps_project_ProfAI.md#phase-2-full-mlops-pipeline-with-docker-compose)*

### CI/CD Pipeline Implementation

**Three-Tier Pipeline Strategy**:
1. **Continuous Integration**: Automated testing, code quality checks, and Docker builds
2. **Continuous Deployment**: Multi-platform deployment to HuggingFace Spaces and container registries
3. **Quality Assurance**: Pylint code analysis and comprehensive test coverage

**Testing Framework**: Comprehensive test suite including:
- **Unit Tests**: Model functionality and data processing validation
- **Integration Tests**: End-to-end API testing and dataset integration

*For CI/CD pipeline details and manual triggers, see [MLOps_project_ProfAI.md](MLOps_project_ProfAI.md#phase-3-cicd-pipeline--model-training)*

### Workflow Orchestration with Airflow

**Automated MLOps Workflows**: Implemented two critical DAGs for production operations:

**Model Evaluation DAG**: Scheduled hourly assessment of model performance across multiple datasets with automated metric recording and comparison reporting.

**Data Drift Detection DAG**: Statistical analysis using Kolmogorov-Smirnov tests to detect distribution changes in input data, with configurable alerting thresholds.

**Production Monitoring**: Integrated workflow results with Prometheus metrics for unified observability across the entire system.

*For workflow configuration and monitoring, see [MLOps_project_ProfAI.md](MLOps_project_ProfAI.md#advanced-features-continuous-monitoring--workflow-orchestration)*

---

## 📊 Metrics & Dashboard Overview

### Comprehensive Metrics Collection

The system implements **multi-layered metrics collection** covering all aspects of the MLOps lifecycle:

**Prediction Metrics**:
- `sentiment_predictions_total`: Counter tracking predictions by sentiment classification
- `sentiment_prediction_latency_seconds`: Histogram of response times with percentile analysis
- `sentiment_text_length`: Distribution of input text lengths for pattern analysis
- `sentiment_confidence`: Model confidence score distributions for quality assessment

**Model Performance Metrics**:
- `model_accuracy`: Accuracy scores by model, dataset, and evaluation split
- `model_f1`, `model_precision`, `model_recall`: Comprehensive performance metrics with class-level granularity
- `model_confusion_matrix`: Detailed classification performance analysis

**System Health Metrics**:
- API response times, error rates, and throughput
- Container resource utilization and system health indicators
- Data drift detection scores and alert frequencies

### Professional Dashboard

**Three Grafana Dashboards** provide comprehensive system observability:

**1. Sentiment Analysis Dashboard**
- **Real-time Prediction Monitoring**: Live charts showing prediction volume and sentiment distribution
- **Performance Tracking**: Response time percentiles and throughput analysis
- **Quality Metrics**: Confidence score distributions and text length analysis

**2. Model Performance Dashboard**
- **Latency Analysis**: Detailed response time breakdowns
- **Throughput Metrics**: Request volume patterns and capacity utilization
- **Error Tracking**: Failed prediction analysis and system reliability metrics

**3. Model Evaluation Metrics Dashboard**
- **Multi-Model Comparison**: Side-by-side performance analysis of base vs fine-tuned models
- **Dataset-Specific Performance**: Evaluation results across TweetEval and Amazon Reviews datasets
- **Class-Level Analysis**: Precision, recall, and F1 scores for positive, negative, and neutral classifications
- **Confusion Matrix Visualization**: Detailed classification performance heatmaps

*For dashboard access and configuration, see [MLOps_project_ProfAI.md](MLOps_project_ProfAI.md#monitoring-stack-overview)*

---

## 🚀 How to Run

For complete setup instructions, testing procedures, and hands-on exploration of all system features, please refer to our comprehensive guide:

**➡️ [MLOps_project_ProfAI.md](MLOps_project_ProfAI.md)**

The guide includes step-by-step instructions for:
- Local development setup and API testing
- Docker Compose deployment with full monitoring stack
- CI/CD pipeline configuration and deployment automation
- Advanced workflow orchestration and monitoring setup

---

## 🔮 EXTRA: Future Enhancements

### RapidAPI Integration for Production Monitoring

**Planned Enhancement**: Integration with **RapidAPI** to create a more realistic production monitoring system that simulates real-world API usage patterns.

**Implementation Strategy**:
- **Real-time Data Ingestion**: Connect to Twitter/social media APIs via RapidAPI for live sentiment analysis
- **Production Traffic Simulation**: Generate realistic user behavior patterns and load testing scenarios
- **Enhanced Monitoring**: Track API usage metrics, rate limiting, and third-party service dependencies

**Technical Implementation**: The system is already designed with microservices architecture that can easily accommodate external API integrations through the existing FastAPI framework and Prometheus monitoring stack.