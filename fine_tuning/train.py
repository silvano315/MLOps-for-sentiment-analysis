import mlflow
import os
import torch
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfFolder
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }

def run_training(
    base_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    dataset_name="mteb/amazon_reviews_multi",
    output_dir="./fine_tuned_model",
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    lora_r=8,
    lora_alpha=16,
    push_to_hub=False,
    hf_token=None,
):
    """Run fine-tuning with LoRA."""

    mlflow.set_experiment("sentiment-analysis-fine-tuning-amazon")

    with mlflow.start_run() as run:
        mlflow.log_param({
            "base_model": base_model,
            "dataset": dataset_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        })
    
    dataset = load_dataset(dataset_name, "en")

    def map_labels(example):
        rating = example['label']
        if rating <= 1:
            # Negative
            example['sentiment'] = 0
        elif rating == 2:
            # Neutral
            example['sentiment'] = 1
        else:
            # Positive
            example['sentiment'] = 2
        return example
        
    dataset = dataset.map(map_labels)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3
    )

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text", "label", "label_text"])

    tokenized_datasets = tokenized_datasets.rename_column("sentiment", "labels")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=push_to_hub,
        report_to="mlflow",
    )

    if push_to_hub and hf_token:
        HfFolder.save_token(hf_token)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    train_results = trainer.train()

    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)

    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)

    mlflow.log_metrics({
        "final_loss": metrics["train_loss"],
        "eval_accuracy": eval_results["eval_accuracy"],
        "eval_f1_macro": eval_results["eval_f1_macro"],
    })

    trainer.save_model(output_dir)

    if push_to_hub:
        model_name = base_model.split("/")[-1]
        repo_name = f"{model_name}-amazon-sentiment"
        trainer.push_to_hub(repo_name)
        mlflow.log_param("hub_model_id", repo_name)

        mlflow.log_artifact(output_dir)
        
    return {
        "run_id": run.info.run_id,
        "metrics": eval_results,
        "model_path": output_dir,
    }

if __name__ == "__main__":
    run_training()