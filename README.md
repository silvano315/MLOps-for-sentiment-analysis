# MLOps for sentiment analysis

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Methods](#methods)
- [Results](#rusults)
- [Key Insights](#key-insights)
- [EXTRA: Real-Time testing with RapidAPI](#extra-real-time-testing-with-rapidapi)
- [How to Run](#how-to-run)

## Project Overview

This repository is the tenth project of the master's degree in AI Engineering with [Profession AI](https://profession.ai), all the credits for the requests and idea go to this team. 

Project steps and information:
1. **Phase 1**: Implement the Sentiment Analysis Model with FastText
    * Model: I need to use a pre-trained FastText model for a sentiment analysis, classifying social texts into positive, neutral or negative sentiment. In fact, I will need to use this specific model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest.
    * Dataset: public datasets containing texts with sentiment labels.
2. **Phase 2**: Build the CI/CD Pipeline
    * ​​CI/CD Pipeline: I need to develop an automated pipeline that manages model training, integration testing and application deployment on HuggingFace.
3. **Phase 3**: Deploy and Continuous Monitoring
    * Deploy on HuggingFace (optional): I could deploy the sentiment analysis model, data and application on HuggingFace, paying attention to integration and scalability.
    * Monitoring System: Set up a monitoring system that continuously evaluates the model's performance and detected sentiment trends.

Features:
- Sentiment analysis of social media texts using RoBERTa
- Datasets from Hugging Face and RapidAPI
- FastAPI-based REST API
- Docker Compose
- Comprehensive monitoring with Grafana
- Workflow orchestration with Airflow
- CI/CD pipeline with GitHub Actions

## Datasets

## Methods

## Results

## Key Insights

## EXTRA: Real-Time testing with RapidAPI

## How to Run