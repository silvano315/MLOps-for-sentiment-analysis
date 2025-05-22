import os
from datetime import datetime, timedelta

from airflow import DAG
#from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.python import PythonOperator
#from airflow.providers.standard.operators.bash import BashOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

dag = DAG(
    dag_id='model_evaluation',
    default_args=default_args,
    description='Evaluate sentiment analysis models on different datasets',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['sentiment', 'evaluation', 'mlops'],
)

def evaluate_base_model_on_tweets(ds, **kwargs):
    """
    Evaluate the base sentiment model on tweet_eval dataset.
    """
    import requests
    import json
    import logging

    logging.info("Starting evaluation of base model on tweet_eval dataset")

    try:
        api_url = Variable.get("sentiment_api_url", "http://api:8080")
        response = requests.post(
            f"{api_url}/api/v1/evaluate",
            params={
                "dataset": "tweet_eval",
                "split": "test",
                "samples": 100,
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest"
            },
            timeout=30
        )

        if response.status_code == 200:
            logging.info("Successfully triggered evaluation of base model on tweet_eval")
            return {
                "status": "success",
                "model": "base",
                "dataset": "tweet_eval",
                "date": ds
            }
        else:
            logging.error(f"Failed to trigger evaluation: {response.text}")
            return {
                "status": "failed",
                "error": response.text
            }
    except Exception as e:
        logging.error(f"Exception during evaluation: {str(e)}")
        raise

def evaluate_finetuned_model_on_tweets(ds, **kwargs):
    """
    Evaluate the fine-tuned sentiment model on tweet_eval dataset.
    """
    import requests
    import json
    import logging
    
    logging.info("Starting evaluation of fine-tuned model on tweet_eval")

    fine_tuned_model = Variable.get(
        "fine_tuned_model_name",
        "silvano315/fine_tuned_model"
    )

    try:
        api_url = Variable.get("sentiment_api_url", "http://api:8080")
        response = requests.post(
            f"{api_url}/api/v1/evaluate",
            params={
                "dataset": "tweet_eval",
                "split": "test",
                "samples": 100,
                "model_name": fine_tuned_model
            },
            timeout=30
        )

        if response.status_code == 200:
            logging.info("Successfully triggered evaluation of fine-tuned model on tweet_eval")
            return {
                "status": "success",
                "model": "finetuned",
                "dataset": "tweet_eval",
                "date": ds
            }
        else:
            logging.error(f"Failed to trigger evaluation: {response.text}")
            return {
                "status": "failed",
                "error": response.text
            }
    except Exception as e:
        logging.error(f"Exception during evaluation: {str(e)}")
        raise

def evaluate_finetuned_model_on_amazon(ds, **kwargs):
    """
    Evaluate the fine-tuned sentiment model on Amazon reviews dataset.
    """
    import requests
    import json
    import logging
    
    logging.info("Starting evaluation of fine-tuned model on Amazon reviews")
    
    # Get the fine-tuned model name from Airflow variables, or use default
    finetuned_model = Variable.get(
        "fine_tuned_model_name",
        "silvano315/fine_tuned_model"
    )

    try:
        api_url = Variable.get("sentiment_api_url", "http://api:8080")
        response = requests.post(
            f"{api_url}/api/v1/evaluate",
            params={
                "dataset": "mteb/amazon_reviews_multi",
                "split": "test",
                "samples": 100,
                "model_name": finetuned_model
            },
            timeout=30
        )
        
        if response.status_code == 200:
            logging.info("Successfully triggered evaluation of fine-tuned model on Amazon reviews")
            return {
                "status": "success",
                "model": "finetuned",
                "dataset": "amazon",
                "date": ds
            }
        else:
            logging.error(f"Failed to trigger evaluation: {response.text}")
            return {
                "status": "failed",
                "error": response.text
            }
    except Exception as e:
        logging.error(f"Exception during evaluation: {str(e)}")
        raise

def generate_comparison_report(ds, **kwargs):
    """
    Generate a comparison report of model evaluations.
    This could involve querying the metrics from Prometheus 
    and creating a summary report.
    """
    import logging
    import requests
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    logging.info("Generating comparison report for model evaluations")

    try:
        prometheus_url = Variable.get("prometheus_url", "http://prometheus:9090")

        base_model_query = 'model_accuracy{model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",dataset="tweet_eval"}'
        finetuned_model_query = 'model_accuracy{model_name="silvano315/fine_tuned_model",dataset="tweet_eval"}'
        finetuned_amazon_query = 'model_accuracy{model_name="silvano315/fine_tuned_model",dataset="mteb/amazon_reviews_multi"}'

        base_resp = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": base_model_query}
        ).json()

        finetuned_resp = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": finetuned_model_query}
        ).json()
        
        finetuned_amazon_resp = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": finetuned_amazon_query}
        ).json()

        try:
            base_accuracy = float(base_resp["data"]["result"][0]["value"][1])
        except (KeyError, IndexError):
            base_accuracy = None
            
        try:
            finetuned_accuracy = float(finetuned_resp["data"]["result"][0]["value"][1])
        except (KeyError, IndexError):
            finetuned_accuracy = None
            
        try:
            finetuned_amazon_accuracy = float(finetuned_amazon_resp["data"]["result"][0]["value"][1])
        except (KeyError, IndexError):
            finetuned_amazon_accuracy = None

        # Basic report, visualization, and markdown generation
        report = f"""
        # Model Evaluation Report: {ds}
        
        ## Accuracy Comparison
        
        - Base model on Tweet data: {f"{base_accuracy:.4f}" if base_accuracy is not None else 'N/A'}
        - Fine-tuned model on Tweet data: {f"{finetuned_accuracy:.4f}" if finetuned_accuracy is not None else 'N/A'}
        - Fine-tuned model on Amazon data: {f"{finetuned_amazon_accuracy:.4f}" if finetuned_amazon_accuracy is not None else 'N/A'}
        
        ## Improvement Analysis
        
        The fine-tuned model shows a {"positive" if finetuned_accuracy and base_accuracy and finetuned_accuracy > base_accuracy else "negative"} 
        impact on tweet data, with a {"gain" if finetuned_accuracy and base_accuracy and finetuned_accuracy > base_accuracy else "loss"} 
        of {f"{abs(finetuned_accuracy - base_accuracy):.4f}" if finetuned_accuracy and base_accuracy else "N/A"} accuracy points.
        
        The fine-tuned model performs {"better" if finetuned_amazon_accuracy and base_accuracy and finetuned_amazon_accuracy > base_accuracy else "worse"} 
        on Amazon data compared to the base model on tweets.
        """

        if all(v is not None for v in [base_accuracy, finetuned_accuracy, finetuned_amazon_accuracy]):
            models = ['Base (Tweets)', 'Fine-tuned (Tweets)', 'Fine-tuned (Amazon)']
            accuracies = [base_accuracy, finetuned_accuracy, finetuned_amazon_accuracy]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
            plt.title(f'Model Accuracy Comparison - {ds}')
            plt.xlabel('Model / Dataset')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            report += f"\n\n## Visualization\n\n![Accuracy Comparison](data:image/png;base64,{plot_base64})\n"
        
        report_path = f"/tmp/model_comparison_{ds.replace('-', '')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logging.info(f"Comparison report saved to {report_path}")
                
        return {
            "status": "success",
            "report_path": report_path,
            "date": ds
        }
        
    except Exception as e:
        logging.error(f"Exception during report generation: {str(e)}")
        raise

# Define and configure the tasks for DAG
evaluate_base_tweets = PythonOperator(
    task_id='evaluate_base_model_on_tweets',
    python_callable=evaluate_base_model_on_tweets,
    provide_context=True,
    dag=dag,
)

evaluate_finetuned_tweets = PythonOperator(
    task_id='evaluate_finetuned_model_on_tweets',
    python_callable=evaluate_finetuned_model_on_tweets,
    provide_context=True,
    dag=dag,
)

evaluate_finetuned_amazon = PythonOperator(
    task_id='evaluate_finetuned_model_on_amazon',
    python_callable=evaluate_finetuned_model_on_amazon,
    provide_context=True,
    dag=dag,
)

wait_for_evaluation = BashOperator(
    task_id="wait_for_evaluation",
    bash_command="sleep 60",
    dag=dag,
)

generate_report = PythonOperator(
    task_id='generate_comparison_report',
    python_callable=generate_comparison_report,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
[evaluate_base_tweets, evaluate_finetuned_tweets, evaluate_finetuned_amazon] >> wait_for_evaluation >> generate_report