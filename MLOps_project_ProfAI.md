# Sentiment Analysis MLOps System - Demo Guide

## 🎯 Project Overview

This repository demonstrates a complete MLOps pipeline for sentiment analysis built with modern tools and best practices. The system classifies social media texts into positive, negative, or neutral sentiment using a fine-tuned RoBERTa model.

## 🏗️ System Architecture

This MLOps system consists of several integrated components:

### 1. Core API Service:
- **FastAPI Application**: RESTful API for sentiment prediction
- **RoBERTa Model**: Pre-trained `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Fine-tuned Model**: Custom model trained on Amazon reviews data
- **Batch Processing**: Support for single and batch predictions

### 2. Infrastructure & Monitoring:
- **Docker Compose**: Containerized deployment with multiple services
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Real-time dashboards and visualization
- **PostgreSQL**: Database for Airflow workflows

### 3. MLOps Pipeline:
- **GitHub Actions**: CI/CD pipeline with automated testing and deployment
- **Apache Airflow**: Workflow orchestration for model evaluation and data drift monitoring
- **HuggingFace Integration**: Model hosting and deployment to HF Spaces

### 4. Data & Evaluation:
- **TweetEval Dataset**: Twitter sentiment data for evaluation
- **Amazon Reviews Dataset**: Multi-language reviews for fine-tuning
- **Automated Evaluation**: Scheduled model performance assessment
- **Data Drift Detection**: Monitoring for distribution changes

---

## 🚀 Setup & Access

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run the application
```bash
uvicorn app.api.main:app --reload --port 8080
```

### 4. Access the Web Interface
- **Main Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs

Opening the main page, you can test the model functionality by writing either a single sentence or a batch of sentences. The app will return the three labels with their respective probabilities and the confidence of the predicted label.

By accessing the `/docs` endpoint, you can verify and test all available GET and POST methods.

---

## 💡 Phase 1: Model Implementation and Sentiment Analysis with FastText using FastAPI

This basic setup represents the **first phase** of this MLOps project. Here, I want to focus on:

### Model Evaluation and Exploration
- Testing the pre-trained RoBERTa model on various text types
- Understanding model behavior with different sentiment expressions
- Evaluating performance on standard datasets (TweetEval, Amazon Reviews)
- Analyzing prediction confidence and probability distributions

### API-First Approach using FastAPI
- Building a robust REST API for model inference 
- Implementing both single and batch prediction capabilities
- Adding comprehensive health checks and monitoring endpoints
- Creating user-friendly interfaces for testing and exploration

### Foundation for MLOps
- Establishing the core service architecture
- Implementing basic monitoring and metrics collection
- Setting up evaluation pipelines for model assessment
- Creating the foundation for more advanced MLOps features

This was a **first attempt** at realizing a complete sentiment analysis system, focusing on model accessibility and basic evaluation capabilities. The simple API and web interface allowed to:

- Test model performance on real-world examples
- Understand the model's strengths and limitations  
- Gather initial metrics and insights
- Validate this approach before building more complex infrastructure

From this foundation, then expanded into full MLOps capabilities with CI/CD pipelines, automated monitoring, workflow orchestration, and production deployment strategies.

### Core Prediction Endpoints

#### `POST /api/v1/sentiment`
**Single text sentiment analysis**
- Input: `{"text": "Your text here"}`
- Output: Sentiment classification with probabilities and confidence score
- Example:
```bash
curl -X POST "http://localhost:8080/api/v1/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
```
- Going to http://localhost:8080/admin/dashboard you can see metrics values are changing.

#### `POST /api/v1/sentiment/batch`
**Batch sentiment analysis**
- Input: `{"texts": ["Text 1", "Text 2", "Text 3"]}`
- Output: Array of sentiment results for each input text
- Example:
```bash
curl -X POST "http://localhost:8080/api/v1/sentiment/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great product!", "Terrible experience", "It'\''s okay"]}'
```

### Model Evaluation Endpoints

#### `POST /api/v1/evaluate`
**Trigger model evaluation on datasets**
- Parameters: `dataset`, `split`, `samples`, `model_name`
- Runs background evaluation on TweetEval or Amazon Reviews datasets
- Records metrics to Prometheus for monitoring
- You can see results in logs, and looking at the next phase you can monitor results with Grafana
- Example:
```bash
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=tweet_eval&split=test&samples=100&model_name=cardiffnlp/twitter-roberta-base-sentiment-latest"
```

```bash
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=mteb/amazon_reviews_multi&split=test&samples=50"
```

If something goes wrong with the Amazon Reviews dataset (e.g. it can't be downloaded), just try to run this:
```bash
pip install -U datasets
```

### Health & Status Endpoints

#### `GET /health`
**Comprehensive health check**
- System status, model availability, resource usage
- GPU availability, memory and disk usage
- Model loading status and error reporting

```bash
curl http://localhost:8080/health
```

#### `GET /ready`
**Readiness probe**
- Quick check if service is ready to serve requests
- Used by orchestration systems

```bash
curl http://localhost:8080/ready
```

#### `GET /live`
**Liveness probe**
- Basic alive check for the service
- Used for health monitoring and auto-restart

```bash
curl http://localhost:8080/admin/metrics
```

### Admin & Monitoring Endpoints

#### `GET /admin/metrics`
**Raw Prometheus metrics**
- Exports all metrics in Prometheus format
- Includes prediction counts, latency, confidence distributions
- Model evaluation metrics (accuracy, F1, precision, recall)
- Used by Prometheus server for scraping

#### `GET /admin/dashboard`
**Simple metrics dashboard**
- Web-based metrics visualization
- Sentiment prediction statistics
- Performance metrics and text analysis
- Real-time metrics without external tools

---

## 🐳 Phase 1 Advanced: Full MLOps Pipeline with Docker Compose

Phase 2 expands this system into a **complete containerized MLOps pipeline** with integrated monitoring, metrics collection, and visualization. This represents the evolution from a simple API to a production-ready system.

### 🏗️ Docker Compose Architecture

Docker Compose setup includes:

- **sentiment-api**: FastAPI application container
- **prometheus**: Metrics collection and storage
- **grafana**: Real-time dashboards and visualization
- **postgres**: Database for workflow management

### 🚀 Running the Complete Stack

#### 1. Build and Start All Services
```bash
docker-compose up --build -d
```

#### 2. Verify All Services Are Running
```bash
# Check container status
docker-compose ps

# Check specific service logs
docker-compose logs -f api
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

### 🌐 Service Access Points

Once the stack is running, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| **API Main Interface** | http://localhost:8080 | Sentiment analysis web UI |
| **API Documentation** | http://localhost:8080/docs | FastAPI auto-generated docs |
| **Health Check** | http://localhost:8080/health | System health status |
| **Prometheus Metrics** | http://localhost:8080/admin/metrics | Raw metrics endpoint |
| **Simple Dashboard** | http://localhost:8080/admin/dashboard | Built-in metrics dashboard |
| **Prometheus Server** | http://localhost:9090 | Prometheus web interface |
| **Grafana Dashboards** | http://localhost:3000 | Professional monitoring (admin/admin) |

### 📊 Monitoring Stack Overview

#### Prometheus (Port 9090)
**Metrics Collection and Storage**
- Scrapes metrics from the API every 15 seconds
- Stores time-series data for analysis
- Provides query interface for metrics exploration
- Configure alerts and rules for monitoring

**Key Metrics Collected:**
- `sentiment_predictions_total`: Count of predictions by sentiment
- `sentiment_prediction_latency_seconds`: Response time metrics
- `sentiment_text_length`: Input text length distribution
- `sentiment_confidence`: Model confidence distribution
- `model_accuracy`: Model evaluation accuracy scores
- `model_f1`, `model_precision`, `model_recall`: Performance metrics

#### Grafana (Port 3000)
**Professional Dashboards and Visualization**
- **Login**: admin/admin (default credentials)
- Pre-configured dashboards for sentiment analysis
- Real-time charts and metrics visualization
- Alerting capabilities for production monitoring

**Available Dashboards:**
1. **Sentiment Analysis Dashboard**: Real-time prediction metrics
2. **Model Performance Dashboard**: Latency and throughput metrics  
3. **Model Evaluation Metrics**: Accuracy, F1, precision, recall by dataset

### 🧪 Testing the Complete Pipeline

#### 1. Generate Sample Traffic
```bash
# Use the provided test script to generate traffic
chmod +x test-sentiment-app.sh
./test-sentiment-app.sh 50 50 20  # 50 positive, 50 negative, 20 neutral

# Or manually send requests
curl -X POST "http://localhost:8080/api/v1/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "This Docker setup is amazing!"}'
```

#### 2. Monitor Metrics in Real-Time

**Check Raw Metrics:**
```bash
# View current metrics
curl http://localhost:8080/admin/metrics | grep sentiment

# Check specific metric
curl http://localhost:8080/admin/metrics | grep sentiment_predictions_total
```

**Prometheus Query Examples:**
```bash
# Total predictions
curl 'http://localhost:9090/api/v1/query?query=sum(sentiment_predictions_total)'

# Prediction rate (per second)
curl 'http://localhost:9090/api/v1/query?query=rate(sentiment_predictions_total%5B5m%5D)'

# Average latency
curl 'http://localhost:9090/api/v1/query?query=rate(sentiment_prediction_latency_seconds_sum%5B5m%5D)%2Frate(sentiment_prediction_latency_seconds_count%5B5m%5D)'
```

#### 3. Explore Grafana Dashboards

1. **Open Grafana**: http://localhost:3000
2. **Login**: admin/admin
3. **Navigate to Dashboards** → Browse
4. **View Pre-configured Dashboards**:
   - Sentiment Analysis Dashboard: Real-time prediction metrics
   - Model Performance: Response times and throughput
   - Model Metrics: Evaluation results and model comparison

### 🔍 Model Evaluation with Monitoring

#### Trigger Evaluation and Watch Metrics
```bash
# Start evaluation on TweetEval dataset and then monitor evaluation progressi in Grafana
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=tweet_eval&split=test&samples=100"
```

#### Compare Model Performance
```bash
# Evaluate base model
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=tweet_eval&split=test&samples=100&model_name=cardiffnlp/twitter-roberta-base-sentiment-latest"

# Evaluate on Amazon reviews (if fine-tuned model available check next section for HuggingFace deployment)
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=mteb/amazon_reviews_multi&split=test&samples=100"
```

### 🛠️ Container Management Commands

#### Useful Docker Compose Commands
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean)
docker-compose down -v

# Restart specific service
docker-compose restart api

# View resource usage
docker stats

# Access container shell
docker-compose exec api bash
docker-compose exec prometheus sh

# Clean Docker system (if needed)
docker system prune -f
```

---

## 🚀 Phase 2: CI/CD Pipeline, Model Fine-tuning & Model Deployment

Phase 2 introduces **automated CI/CD pipelines** with model training, testing, and deployment to multiple platforms. This represents the full MLOps lifecycle automation.

### 🏗️ CI/CD Architecture Overview

This automated pipeline includes:

- **Model Fine-tuning**: PEFT/LoRA training with MLflow tracking
- **Automated Testing**: Unit and integration tests with pytest
- **GitHub Actions**: CI/CD workflows for testing and deployment
- **Multi-platform Deployment**: HuggingFace Spaces, DockerHub, and model hosting
- **Model Registry**: Versioned models on HuggingFace Hub

---

## 🧠 Model Fine-tuning with PEFT & MLflow

### Fine-tuning Implementation

Our fine-tuning process uses **PEFT (Parameter Efficient Fine-Tuning)** with **LoRA (Low-Rank Adaptation)** for efficient training:

**Location**: `fine_tuning/train.py`

#### Key Features:
- **Base Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Dataset**: Amazon Reviews Multi (English subset)
- **Technique**: LoRA with rank=8, alpha=16
- **Tracking**: MLflow for experiment management

#### Run Fine-tuning Locally:
```bash
cd fine_tuning/

# Start MLflow tracking (optional)
mlflow ui --port 5000 &

# Run fine-tuning
python train.py

# Customize parameters
python -c "
from train import run_training
run_training(
    num_epochs=5,
    batch_size=16,
    learning_rate=3e-5,
    split_ratio=0.2,
    push_to_hub=True,
    hf_token='HF_TOKEN'
)
"
```

#### View Training Progress:
```bash
open http://localhost:5000

# Check MLFlow logs
ls mlruns/
```

---

## 🧪 Testing Framework

### Unit Tests (`tests/unit/`)

#### Model Tests (`test_model.py`):

What you can test:
- Model initialization and loading
- Sentiment prediction accuracy on known examples
- Confidence score validation
- Probability distribution correctness

```bash
pytest tests/unit/test_model.py -v

# Test specific functions
pytest tests/unit/test_model.py::test_sentiment_prediction -v
pytest tests/unit/test_model.py::test_confidence_matches_probs -v
```

#### Dataset Tests (`test_datasets.py`):

What you can test:
- Dataset registry configuration
- Amazon reviews star mapping (0-4 → negative/neutral/positive)
- Dataset configuration retrieval

```bash
pytest tests/unit/test_datasets.py -v
```

### Integration Tests (`tests/integration/`)

What you can test:
- All API endpoints functionality
- Request/response validation
- Error handling for invalid inputs
- Health check endpoints
- Batch processing capabilities

#### API Integration Tests (`test_api.py`):
```bash
pytest tests/integration/test_api.py -v

# Test specific endpoints
pytest tests/integration/test_api.py::test_sentiment_analysis_endpoint -v
pytest tests/integration/test_api.py::test_batch_sentiment_analysis_endpoint -v
```

#### Dataset Integration Tests (`test_dataset_integration.py`):

What you can test:
- Dataset download and preprocessing
- Model predictions on real dataset samples
- End-to-end data flow validation

```bash
pytest tests/integration/test_dataset_integration.py -v
```

---

## ⚙️ GitHub Actions CI/CD Pipeline

### Pipeline Overview

The CI/CD consists of **3 automated workflows**:

**Location**: `.github/workflows/`

1. **`ci.yml`** - Continuous Integration (Testing)
2. **`cd.yml`** - Continuous Deployment  
3. **`pylint.yml`** - Code Quality Checks

### 1. CI Pipeline (`.github/workflows/ci.yml`)

#### What it does:
```yaml
# Workflow steps:
1. Setup Python 3.10
2. Install dependencies
3. Code formatting checks (black, isort)
4. Type checking (mypy)
5. Run unit tests
6. Run integration tests
7. Build Docker image
8. Test Docker container health
```

### 2. CD Pipeline (`.github/workflows/cd.yml`)

#### What it does:
```yaml
# Deployment steps:
1. Build Docker image with metadata
2. Push to GitHub Container Registry (ghcr.io)
3. Push to DockerHub (optional)
4. Deploy App to HuggingFace Spaces
5. Update production environment
```

### 3. Code Quality Pipeline (`.github/workflows/pylint.yml`)

**What it does**:
- Runs pylint on all Python code
- Enforces code quality standards (minimum score: 8.0/10)
- Checks app/, tests/, and scripts/ directories

### Workflow Results:

#### GitHub UI:
1. Go to your repository on GitHub
2. Click **"Actions"** tab
3. View workflow runs and results
4. Click on specific runs for detailed logs

---

## 🚢 Multi-Platform Deployment

### 1. HuggingFace Spaces Deployment

#### Automatic Deployment via CD Pipeline:

**Configuration**: The CD pipeline automatically deploys to HF Spaces using:
- **Script**: `scripts/deploy_to_huggingface.py`
- **Docker Image**: Built and pushed to registry
- **Environment Variables**: Be carefull, there are Secrets to set up and configure

#### Manual Deployment:
```bash
export HF_TOKEN="your_hf_token"
export HF_USERNAME="your_username"
export HF_SPACE_NAME="sentiment-analysis-api"
export DOCKER_IMAGE="ghcr.io/your_username/sentiment-api:latest"

python scripts/deploy_to_huggingface.py
```

#### Access Deployed App:
- **Live Demo**: https://huggingface.co/spaces/silvano315/sentiment-analysis-api

### 2. DockerHub Deployment

You need to set up and configure Secrets also for DockerHub.

#### Automatic Push via CD Pipeline:

The pipeline builds and pushes Docker images to multiple registries:

```bash
# GitHub Container Registry
docker pull ghcr.io/your_username/sentiment-api:latest

# DockerHub (if configured)  
docker pull your_username/sentiment-api:latest
```

### 3. Fine-tuned Model Deployment

#### Model Available on HuggingFace Hub:

Here I give you my fine-tuned model, but you can also use your own fine-tuned model. Just change the model_name.

**Pre-trained Fine-tuned Model**: https://huggingface.co/silvano315

#### Using the Fine-tuned Model:

```python
from app.models.model_loader import SentimentModel
from app.models.prediction import SentimentPredictor

model = SentimentModel(model_name="silvano315/fine_tuned_model")
predictor = SentimentPredictor(model)
predictor.predict("This fine-tuned model works great!")

# You can also set environment variable to use fine-tuned model
import os
os.environ['MODEL_NAME'] = 'silvano315/fine_tuned_model'

import requests
response = requests.post(
    "http://localhost:8080/api/v1/sentiment",
    json={"text": "This fine-tuned model works great!"}
)
```

#### Compare Models Performance:
```bash
# Evaluate base model
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=tweet_eval&split=test&samples=100&model_name=cardiffnlp/twitter-roberta-base-sentiment-latest"

# Evaluate fine-tuned model  
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=tweet_eval&samples=100&model_name=silvano315/fine_tuned_model"

# Evaluate fine-tuned model on Amazon reviews
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=mteb/amazon_reviews_multi&samples=100&model_name=silvano315/fine_tuned_model"
```

---

## 🔬 Phase 3: Continuous Monitoring & Workflow Orchestration with Airflow

This final phase introduces **advanced MLOps capabilities** with Airflow for workflow orchestration, automated model evaluation, and data drift detection for production monitoring.

### 🌊 Apache Airflow Integration

#### Airflow Architecture

Airflow setup includes:
- **Airflow Webserver**: Web UI for workflow management (Port 8088)
- **Airflow Scheduler**: Task scheduling and execution
- **PostgreSQL**: Airflow metadata database
- **Custom DAGs**: Automated workflows for ML operations

#### Starting Airflow with Docker Compose

```bash
docker-compose up --build -d

# Verify Airflow services
docker-compose ps | grep airflow

# Check Airflow logs
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler
```

#### Access Airflow Web Interface

- **Airflow UI**: http://localhost:8088
- **Default Login**: admin/admin (configured in docker-compose.yml)
- **DAGs Location**: `infrastructure/airflow/dags/`

---

## 📊 Automated Model Evaluation Workflow

### Model Evaluation DAG

**Location**: `infrastructure/airflow/dags/model_evaluation_dag.py`

#### DAG Tasks:
1. **evaluate_base_model_on_tweets**: Test base RoBERTa on TweetEval
2. **evaluate_finetuned_model_on_tweets**: Test fine-tuned model on TweetEval  
3. **evaluate_finetuned_model_on_amazon**: Test fine-tuned model on Amazon Reviews
4. **wait_for_evaluation**: Buffer time for evaluation completion
5. **generate_comparison_report**: Create performance comparison report

#### Manual DAG Execution:

```bash
# Access Airflow UI at http://localhost:8088
# Navigate to DAGs → model_evaluation
# Click "Trigger DAG" to run manually

# Or via Airflow CLI in container
docker-compose exec airflow-webserver airflow dags trigger model_evaluation
```

#### View Evaluation Results:

Three different methods to check evaluation results.

```bash
# Prometheus
curl 'http://localhost:9090/api/v1/query?query=model_accuracy'

# Grafana Model Metrics Dashboard
# Open http://localhost:3000 → Dashboards → Model Evaluation Metrics

# Reports in container
docker-compose exec airflow-webserver ls /tmp/model_comparison_*
```

---

## 🚨 Data Drift Monitoring

### Data Drift Detection DAG

**Location**: `infrastructure/airflow/dags/data_drift_monitoring_dag.py`

#### DAG Tasks:
1. **sample_reference_data**: Collect baseline data samples
2. **sample_current_data**: Collect current data samples
3. **calculate_performance_drift**: Compare model performance metrics
4. **generate_drift_report**: Create comprehensive drift analysis report
5. **send_alerts_if_needed**: Generate alerts if drift detected

#### Manual Drift Detection:

```bash
# Trigger drift monitoring manually
# In Airflow UI: DAGs → data_drift_monitoring → Trigger DAG

# Or via CLI
docker-compose exec airflow-webserver airflow dags trigger data_drift_monitoring
```

#### View Drift Reports:

```bash
docker-compose exec airflow-webserver ls /tmp/drift_*

docker-compose exec airflow-webserver cat /tmp/drift_monitoring_report_$(date +%Y%m%d).md

docker-compose exec airflow-webserver ls /tmp/drift_report_*.png

docker-compose exec airflow-webserver cat /tmp/drift_alerts_$(date +%Y%m%d).json
```

---