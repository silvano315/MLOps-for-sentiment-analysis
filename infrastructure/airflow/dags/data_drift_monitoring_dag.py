import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
#from airflow.providers.standard.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

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
    dag_id='data_drift_monitoring',
    default_args=default_args,
    description='Monitor data drift in sentiment analysis models',
    schedule_interval=timedelta(hours=12),
    catchup=False,
    tags=['sentiment', 'drift', 'monitoring', 'mlops'],
)

def get_drift_config():
    """Get drift monitoring configuration from Airflow variables."""
    
    # No Drift Monitoring configuration
    default_config = {
        "reference_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "current_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "reference_dataset": "tweet_eval",
        "current_dataset": "tweet_eval",
        "reference_samples": 200,
        "current_samples": 20
    }
    
    try:
        config = {
            "reference_model": Variable.get("drift_reference_model", default_config["reference_model"]),
            "current_model": Variable.get("drift_current_model", default_config["current_model"]),
            "reference_dataset": Variable.get("drift_reference_dataset", default_config["reference_dataset"]),
            "current_dataset": Variable.get("drift_current_dataset", default_config["current_dataset"]),
            "reference_samples": int(Variable.get("drift_reference_samples", default_config["reference_samples"])),
            "current_samples": int(Variable.get("drift_current_samples", default_config["current_samples"]))
        }
    except Exception as e:
        logging.warning(f"Error getting drift config from Variables: {e}. Using default config.")
        config = default_config
    
    return config

def get_alert_thresholds():
    """Get alert thresholds from Airflow variables."""
    
    default_thresholds = {
        "text_length_ks_pvalue": 0.05,
        "accuracy_drop": 0.10,
        "f1_drop": 0.08,
        "confidence_shift": 0.15
    }
    
    try:
        thresholds = {
            "text_length_ks_pvalue": float(Variable.get("alert_text_length_ks", default_thresholds["text_length_ks_pvalue"])),
            "accuracy_drop": float(Variable.get("alert_accuracy_drop", default_thresholds["accuracy_drop"])),
            "f1_drop": float(Variable.get("alert_f1_drop", default_thresholds["f1_drop"])),
            "confidence_shift": float(Variable.get("alert_confidence_shift", default_thresholds["confidence_shift"]))
        }
    except Exception as e:
        logging.warning(f"Error getting alert thresholds from Variables: {e}. Using default thresholds.")
        thresholds = default_thresholds
    
    return thresholds

def sample_reference_data(ds, **kwargs):
    """Sample reference data for drift comparison."""
    import requests
    import random
    
    logging.info("Sampling reference data for drift monitoring")
    
    config = get_drift_config()
    
    try:
        from data.datasets.download_datasets import download_and_prepare_datasets
        from data.datasets.dataset_registry import get_dataset_config
        
        datasets = download_and_prepare_datasets([config["reference_dataset"]])
        dataset_dict = datasets[config["reference_dataset"]]
        dataset = dataset_dict["dataset"]
        
        dataset_config = get_dataset_config(config["reference_dataset"])
        text_column = dataset_config.get("text_column", "text")
        label_column = dataset_config.get("label_column", "label")
        
        split = "test" if "test" in dataset else "train"
        data_split = dataset[split]
        
        total_samples = len(data_split)
        sample_indices = random.sample(range(total_samples), min(config["reference_samples"], total_samples))
        sampled_data = data_split.select(sample_indices)
        
        reference_data = {
            "texts": sampled_data[text_column],
            "labels": sampled_data[label_column] if label_column in sampled_data.features else [],
            "text_lengths": [len(text) for text in sampled_data[text_column]],
            "dataset": config["reference_dataset"],
            "model": config["reference_model"],
            "sample_size": len(sampled_data),
            "timestamp": ds
        }
        
        reference_file = f"/tmp/reference_data_{ds.replace('-', '')}.json"
        with open(reference_file, 'w') as f:
            json.dump(reference_data, f, default=str)
        
        logging.info(f"Reference data sampled: {len(sampled_data)} samples from {config['reference_dataset']}")
        logging.info(f"Reference data saved to {reference_file}")
        
        return {
            "status": "success",
            "dataset": config["reference_dataset"],
            "samples": len(sampled_data),
            "file_path": reference_file
        }
        
    except Exception as e:
        logging.error(f"Error sampling reference data: {str(e)}")
        raise

def sample_current_data(ds, **kwargs):
    """Sample current data for drift comparison."""
    import requests
    import random
    
    logging.info("Sampling current data for drift monitoring")
    
    config = get_drift_config()
    
    try:
        from data.datasets.download_datasets import download_and_prepare_datasets
        from data.datasets.dataset_registry import get_dataset_config
        
        datasets = download_and_prepare_datasets([config["current_dataset"]])
        dataset_dict = datasets[config["current_dataset"]]
        dataset = dataset_dict["dataset"]
        
        dataset_config = get_dataset_config(config["current_dataset"])
        text_column = dataset_config.get("text_column", "text")
        label_column = dataset_config.get("label_column", "label")
        
        split = "test" if "test" in dataset else "train"
        data_split = dataset[split]
        
        total_samples = len(data_split)
        sample_indices = random.sample(range(total_samples), min(config["current_samples"], total_samples))
        sampled_data = data_split.select(sample_indices)
        
        current_data = {
            "texts": sampled_data[text_column],
            "labels": sampled_data[label_column] if label_column in sampled_data.features else [],
            "text_lengths": [len(text) for text in sampled_data[text_column]],
            "dataset": config["current_dataset"],
            "model": config["current_model"],
            "sample_size": len(sampled_data),
            "timestamp": ds
        }
        
        current_file = f"/tmp/current_data_{ds.replace('-', '')}.json"
        with open(current_file, 'w') as f:
            json.dump(current_data, f, default=str)
        
        logging.info(f"Current data sampled: {len(sampled_data)} samples from {config['current_dataset']}")
        logging.info(f"Current data saved to {current_file}")
        
        return {
            "status": "success",
            "dataset": config["current_dataset"],
            "samples": len(sampled_data),
            "file_path": current_file
        }
        
    except Exception as e:
        logging.error(f"Error sampling current data: {str(e)}")
        raise

def calculate_performance_drift(ds, **kwargs):
    """Calculate performance drift between reference and current models/datasets."""
    import requests
    
    logging.info("Calculating performance drift")
    
    config = get_drift_config()
    api_url = Variable.get("sentiment_api_url", "http://api:8080")
    
    try:
        reference_file = f"/tmp/reference_data_{ds.replace('-', '')}.json"
        current_file = f"/tmp/current_data_{ds.replace('-', '')}.json"
        
        with open(reference_file, 'r') as f:
            reference_data = json.load(f)
        
        with open(current_file, 'r') as f:
            current_data = json.load(f)
        
        logging.info(f"Evaluating {config['reference_model']} on {config['reference_dataset']}")
        ref_response = requests.post(
            f"{api_url}/api/v1/evaluate",
            params={
                "dataset": config["reference_dataset"],
                "split": "test",
                "samples": config["reference_samples"],
                "model_name": config["reference_model"]
            },
            timeout=300
        )
        
        import time
        time.sleep(30)
        
        logging.info(f"Evaluating {config['current_model']} on {config['current_dataset']}")
        curr_response = requests.post(
            f"{api_url}/api/v1/evaluate",
            params={
                "dataset": config["current_dataset"],
                "split": "test", 
                "samples": config["current_samples"],
                "model_name": config["current_model"]
            },
            timeout=300
        )
        
        time.sleep(30)
        
        prometheus_url = Variable.get("prometheus_url", "http://prometheus:9090")
        
        ref_acc_query = f'model_accuracy{{model_name="{config["reference_model"]}",dataset="{config["reference_dataset"]}"}}'
        ref_f1_query = f'model_f1{{model_name="{config["reference_model"]}",dataset="{config["reference_dataset"]}",class_label="macro",average="macro"}}'
        
        curr_acc_query = f'model_accuracy{{model_name="{config["current_model"]}",dataset="{config["current_dataset"]}"}}'
        curr_f1_query = f'model_f1{{model_name="{config["current_model"]}",dataset="{config["current_dataset"]}",class_label="macro",average="macro"}}'
        
        # Metrics from Prometheus
        def get_metric_value(query):
            try:
                response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": query})
                result = response.json()
                if result["data"]["result"]:
                    return float(result["data"]["result"][0]["value"][1])
                return None
            except:
                return None
        
        ref_accuracy = get_metric_value(ref_acc_query)
        ref_f1 = get_metric_value(ref_f1_query)
        curr_accuracy = get_metric_value(curr_acc_query)
        curr_f1 = get_metric_value(curr_f1_query)
        
        # Drift metrics
        performance_drift = {
            "reference_accuracy": ref_accuracy,
            "current_accuracy": curr_accuracy,
            "reference_f1": ref_f1,
            "current_f1": curr_f1,
            "accuracy_drift": (curr_accuracy - ref_accuracy) if (ref_accuracy and curr_accuracy) else None,
            "f1_drift": (curr_f1 - ref_f1) if (ref_f1 and curr_f1) else None,
            "timestamp": ds
        }
        
        # Length drift
        ref_lengths = reference_data["text_lengths"]
        curr_lengths = current_data["text_lengths"]
        
        # Kolmogorov-Smirnov test for text length distributions
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_lengths, curr_lengths)
        
        length_drift = {
            "reference_mean_length": np.mean(ref_lengths),
            "current_mean_length": np.mean(curr_lengths),
            "reference_std_length": np.std(ref_lengths),
            "current_std_length": np.std(curr_lengths),
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "length_drift_detected": ks_pvalue < get_alert_thresholds()["text_length_ks_pvalue"]
        }
        
        drift_results = {
            "performance_drift": performance_drift,
            "length_drift": length_drift,
            "config": config,
            "timestamp": ds
        }
        
        drift_file = f"/tmp/drift_results_{ds.replace('-', '')}.json"
        with open(drift_file, 'w') as f:
            json.dump(drift_results, f, default=str)
        
        logging.info(f"Performance drift calculated and saved to {drift_file}")
        logging.info(f"Accuracy drift: {performance_drift.get('accuracy_drift', 'N/A')}")
        logging.info(f"F1 drift: {performance_drift.get('f1_drift', 'N/A')}")
        logging.info(f"Text length KS p-value: {ks_pvalue:.4f}")
        
        return {
            "status": "success",
            "drift_file": drift_file,
            "accuracy_drift": performance_drift.get("accuracy_drift"),
            "f1_drift": performance_drift.get("f1_drift"),
            "length_drift_detected": length_drift["length_drift_detected"]
        }
        
    except Exception as e:
        logging.error(f"Error calculating performance drift: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_drift_report(ds, **kwargs):
    """Generate comprehensive drift monitoring report."""
    
    logging.info("Generating drift monitoring report")
    
    try:
        drift_file = f"/tmp/drift_results_{ds.replace('-', '')}.json"
        reference_file = f"/tmp/reference_data_{ds.replace('-', '')}.json"
        current_file = f"/tmp/current_data_{ds.replace('-', '')}.json"
        
        with open(drift_file, 'r') as f:
            drift_results = json.load(f)
        
        with open(reference_file, 'r') as f:
            reference_data = json.load(f)
            
        with open(current_file, 'r') as f:
            current_data = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Data Drift Monitoring Report - {ds}', fontsize=16)
        
        # Text Length Distributions
        axes[0, 0].hist(reference_data["text_lengths"], alpha=0.7, label='Reference', bins=20, color='blue')
        axes[0, 0].hist(current_data["text_lengths"], alpha=0.7, label='Current', bins=20, color='red')
        axes[0, 0].set_xlabel('Text Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Text Length Distribution Comparison')
        axes[0, 0].legend()
        
        # Performance Comparison
        performance = drift_results["performance_drift"]
        metrics = ['Accuracy', 'F1 Score']
        ref_values = [performance.get("reference_accuracy", 0), performance.get("reference_f1", 0)]
        curr_values = [performance.get("current_accuracy", 0), performance.get("current_f1", 0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, ref_values, width, label='Reference', color='blue', alpha=0.7)
        axes[0, 1].bar(x + width/2, curr_values, width, label='Current', color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Performance Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # Drift Summary
        length_drift = drift_results["length_drift"]
        drift_metrics = ['Mean Length', 'Std Length']
        ref_length_values = [length_drift["reference_mean_length"], length_drift["reference_std_length"]]
        curr_length_values = [length_drift["current_mean_length"], length_drift["current_std_length"]]
        
        x = np.arange(len(drift_metrics))
        axes[1, 0].bar(x - width/2, ref_length_values, width, label='Reference', color='blue', alpha=0.7)
        axes[1, 0].bar(x + width/2, curr_length_values, width, label='Current', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Text Length Statistics')
        axes[1, 0].set_ylabel('Characters')
        axes[1, 0].set_title('Text Length Statistics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(drift_metrics)
        axes[1, 0].legend()
        
        # Alert Summary
        axes[1, 1].axis('off')
        
        alert_text = f"""
DRIFT MONITORING SUMMARY

Configuration:
- Reference: {drift_results['config']['reference_model']} on {drift_results['config']['reference_dataset']}
- Current: {drift_results['config']['current_model']} on {drift_results['config']['current_dataset']}

Performance Drift:
- Accuracy: {performance.get('accuracy_drift', 'N/A'):.4f if performance.get('accuracy_drift') else 'N/A'}
- F1 Score: {performance.get('f1_drift', 'N/A'):.4f if performance.get('f1_drift') else 'N/A'}

Text Length Drift:
- KS Test p-value: {length_drift['ks_pvalue']:.4f}
- Drift detected: {'YES' if length_drift['length_drift_detected'] else 'NO'}

Sample Sizes:
- Reference: {reference_data['sample_size']} samples
- Current: {current_data['sample_size']} samples
        """
        
        axes[1, 1].text(0.1, 0.9, alert_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        plot_file = f"/tmp/drift_report_{ds.replace('-', '')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Markdown report
        markdown_report = f"""
# Data Drift Monitoring Report
**Date**: {ds}

## Configuration
- **Reference Model**: {drift_results['config']['reference_model']}
- **Reference Dataset**: {drift_results['config']['reference_dataset']} ({reference_data['sample_size']} samples)
- **Current Model**: {drift_results['config']['current_model']}
- **Current Dataset**: {drift_results['config']['current_dataset']} ({current_data['sample_size']} samples)

## Performance Drift Analysis

| Metric | Reference | Current | Drift |
|--------|-----------|---------|-------|
| Accuracy | {performance.get('reference_accuracy', 'N/A'):.4f if performance.get('reference_accuracy') else 'N/A'} | {performance.get('current_accuracy', 'N/A'):.4f if performance.get('current_accuracy') else 'N/A'} | {performance.get('accuracy_drift', 'N/A'):.4f if performance.get('accuracy_drift') else 'N/A'} |
| F1 Score | {performance.get('reference_f1', 'N/A'):.4f if performance.get('reference_f1') else 'N/A'} | {performance.get('current_f1', 'N/A'):.4f if performance.get('current_f1') else 'N/A'} | {performance.get('f1_drift', 'N/A'):.4f if performance.get('f1_drift') else 'N/A'} |

## Text Length Distribution Analysis

| Statistic | Reference | Current |
|-----------|-----------|---------|
| Mean Length | {length_drift['reference_mean_length']:.2f} | {length_drift['current_mean_length']:.2f} |
| Std Length | {length_drift['reference_std_length']:.2f} | {length_drift['current_std_length']:.2f} |
| KS Test p-value | {length_drift['ks_pvalue']:.4f} |
| Drift Detected | {'YES ⚠️' if length_drift['length_drift_detected'] else 'NO ✅'} |

## Alert Summary
{f"🚨 **TEXT LENGTH DRIFT DETECTED** - KS test p-value ({length_drift['ks_pvalue']:.4f}) below threshold ({get_alert_thresholds()['text_length_ks_pvalue']})" if length_drift['length_drift_detected'] else "✅ No significant text length drift detected"}

## Visualization
![Drift Report]({plot_file})
        """
        
        report_file = f"/tmp/drift_monitoring_report_{ds.replace('-', '')}.md"
        with open(report_file, 'w') as f:
            f.write(markdown_report)
        
        logging.info(f"Drift monitoring report generated: {report_file}")
        logging.info(f"Drift monitoring plot saved: {plot_file}")
        
        return {
            "status": "success",
            "report_file": report_file,
            "plot_file": plot_file,
            "length_drift_detected": length_drift["length_drift_detected"]
        }
        
    except Exception as e:
        logging.error(f"Error generating drift report: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def send_alerts_if_needed(ds, **kwargs):
    """Send alerts if drift is detected based on configured thresholds."""
    
    logging.info("Checking for drift alerts")
    
    try:
        drift_file = f"/tmp/drift_results_{ds.replace('-', '')}.json"
        with open(drift_file, 'r') as f:
            drift_results = json.load(f)
        
        config = drift_results["config"]
        performance_drift = drift_results["performance_drift"]
        length_drift = drift_results["length_drift"]
        thresholds = get_alert_thresholds()
        
        alerts = []
        
        if length_drift["length_drift_detected"]:
            alerts.append({
                "type": "TEXT_LENGTH_DRIFT",
                "severity": "WARNING",
                "message": f"Text length distribution drift detected (KS p-value: {length_drift['ks_pvalue']:.4f})",
                "threshold": thresholds["text_length_ks_pvalue"],
                "actual_value": length_drift["ks_pvalue"]
            })
        
        if performance_drift.get("accuracy_drift") is not None:
            if abs(performance_drift["accuracy_drift"]) > thresholds["accuracy_drop"]:
                alerts.append({
                    "type": "ACCURACY_DRIFT",
                    "severity": "CRITICAL" if performance_drift["accuracy_drift"] < -thresholds["accuracy_drop"] else "WARNING",
                    "message": f"Accuracy drift detected: {performance_drift['accuracy_drift']:.4f}",
                    "threshold": thresholds["accuracy_drop"],
                    "actual_value": abs(performance_drift["accuracy_drift"])
                })
        
        if performance_drift.get("f1_drift") is not None:
            if abs(performance_drift["f1_drift"]) > thresholds["f1_drop"]:
                alerts.append({
                    "type": "F1_DRIFT",
                    "severity": "CRITICAL" if performance_drift["f1_drift"] < -thresholds["f1_drop"] else "WARNING",
                    "message": f"F1 score drift detected: {performance_drift['f1_drift']:.4f}",
                    "threshold": thresholds["f1_drop"],
                    "actual_value": abs(performance_drift["f1_drift"])
                })
        
        if alerts:
            logging.warning(f"🚨 DRIFT ALERTS DETECTED ({len(alerts)} alerts)")
            for alert in alerts:
                logging.warning(f"[{alert['severity']}] {alert['type']}: {alert['message']}")
        else:
            logging.info("✅ No drift alerts detected - all metrics within acceptable thresholds")
        
        alerts_file = f"/tmp/drift_alerts_{ds.replace('-', '')}.json"
        with open(alerts_file, 'w') as f:
            json.dump({
                "timestamp": ds,
                "config": config,
                "alerts": alerts,
                "alert_count": len(alerts)
            }, f, default=str)
        
        logging.info(f"Alert summary saved to {alerts_file}")
        
        return {
            "status": "success",
            "alerts_detected": len(alerts) > 0,
            "alert_count": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logging.error(f"Error checking drift alerts: {str(e)}")
        raise


# TASK & DEPENDENCES

sample_reference = PythonOperator(
    task_id='sample_reference_data',
    python_callable=sample_reference_data,
    provide_context=True,
    dag=dag,
)

sample_current = PythonOperator(
    task_id='sample_current_data',
    python_callable=sample_current_data,
    provide_context=True,
    dag=dag,
)

calculate_drift = PythonOperator(
    task_id='calculate_performance_drift',
    python_callable=calculate_performance_drift,
    provide_context=True,
    dag=dag,
)

generate_report = PythonOperator(
    task_id='generate_drift_report',
    python_callable=generate_drift_report,
    provide_context=True,
    dag=dag,
)

send_alerts = PythonOperator(
    task_id='send_alerts_if_needed',
    python_callable=send_alerts_if_needed,
    provide_context=True,
    dag=dag,
)

[sample_reference, sample_current] >> calculate_drift >> generate_report >> send_alerts