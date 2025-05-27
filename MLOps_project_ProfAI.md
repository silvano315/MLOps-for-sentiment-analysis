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

#### `POST /api/v1/sentiment/batch`
**Batch sentiment analysis**
- Input: `{"texts": ["Text 1", "Text 2", "Text 3"]}`
- Output: Array of sentiment results for each input text
- Example:
```bash
curl -X POST "http://localhost:8080/api/v1/sentiment/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great product!", "Terrible experience", "It\'s okay"]}'
```

### Model Evaluation Endpoints

#### `POST /api/v1/evaluate`
**Trigger model evaluation on datasets**
- Parameters: `dataset`, `split`, `samples`, `model_name`
- Runs background evaluation on TweetEval or Amazon Reviews datasets
- Records metrics to Prometheus for monitoring
- Example:
```bash
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=tweet_eval&split=test&samples=100&model_name=cardiffnlp/twitter-roberta-base-sentiment-latest"
```

```bash
curl -X POST "http://localhost:8080/api/v1/evaluate?dataset=mteb/amazon_reviews_multi&split=test&samples=50"
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
- Used by orchestration systems (Kubernetes, Docker Compose)

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

## 🎯 Next Steps

Once you've explored Phase 1, you can continue with:

- **Phase 2**: Full MLOps pipeline with Docker Compose, monitoring stack
- **Phase 3**: CI/CD integration, automated deployments, workflow orchestration
- **Advanced Features**: Data drift detection, model comparison, production monitoring

Each phase builds upon the previous one, creating a comprehensive MLOps system for production-ready sentiment analysis.