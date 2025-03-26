import os
import random
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from pymongo import MongoClient
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score
from shutil import rmtree  # Import the rmtree function

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EmailClassifierTraining")

# -----------------------------------------------------------------------------
# Reproducibility: Set seeds
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Global definitions
# -----------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2

# -----------------------------------------------------------------------------
# MongoDB Operations: Load emails
# -----------------------------------------------------------------------------


def load_emails_from_mongo(
    uri: str = "mongodb://localhost:27017",
    db_name: str = "email_routing",
    collection_name: str = "email_datasets",
) -> List[Dict[str, Any]]:
    """
    Connect to MongoDB and retrieve email documents.
    """
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        emails = list(collection.find({}))
        logger.info(f"Retrieved {len(emails)} emails from MongoDB.")
        return emails
    except Exception as e:
        logger.error(f"Error retrieving emails: {e}")
        raise

# -----------------------------------------------------------------------------
# Data Processing: Assign labels and combine text fields
# -----------------------------------------------------------------------------


def process_emails(emails: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Process emails: combine subject and body and assign label:
       - "update" if 'is_update_case' is True
       - "request" otherwise.
    """
    label_mapping = {"update": 0, "request": 1}
    texts: List[str] = []
    labels: List[int] = []
    update_count = 0
    request_count = 0

    for email in emails:
        if email.get("is_update_case", False):
            label_str = "update"
            update_count += 1
        else:
            label_str = "request"
            request_count += 1

        subject = email.get("subject", "").strip()
        body = email.get("body", "").strip()
        combined_text = f"{subject} {body}".strip()
        texts.append(combined_text)
        labels.append(label_mapping[label_str])

        logger.info("Email subject: '%s' categorized as '%s'.",
                    subject, label_str)

    logger.info("Categorization summary: update=%d, request=%d",
                update_count, request_count)
    return {"text": texts, "label": labels}

# -----------------------------------------------------------------------------
# Tokenization: Create tokenized dataset
# -----------------------------------------------------------------------------


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 128) -> Dataset:
    """
    Tokenize a Dataset using the provided tokenizer.
    """
    def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        processed_texts = [
            txt if txt.strip() else "No text available." for txt in batch["text"]]
        return tokenizer(
            processed_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns("text")
    logger.info("Dataset tokenization complete.")
    return tokenized_dataset

# -----------------------------------------------------------------------------
# Compute evaluation metrics
# -----------------------------------------------------------------------------


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# -----------------------------------------------------------------------------
# Heuristic reasoning for misclassifications
# -----------------------------------------------------------------------------


def get_label_name(label: int) -> str:
    return "update" if label == 0 else "request"


def heuristic_reason(text: str, actual: int, predicted: int) -> str:
    text_lower = text.lower()
    reasons = []
    if actual == 0 and "update" not in text_lower:
        reasons.append("Missing 'update' keyword.")
    elif actual == 1 and "update" in text_lower:
        reasons.append("Contains 'update' keyword unexpectedly.")
    if len(text.strip()) < 20:
        reasons.append("Text is very short and ambiguous.")
    return ", ".join(reasons) if reasons else "No obvious reason."

# -----------------------------------------------------------------------------
# Model initialization function required for hyperparameter search
# -----------------------------------------------------------------------------


def model_init() -> AutoModelForSequenceClassification:
    """
    Function to instantiate a new model.
    Used during hyperparameter search to get fresh model instances.
    """
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# -----------------------------------------------------------------------------
# Optional: Hyperparameter search with Trainer
# -----------------------------------------------------------------------------


def run_hyperparameter_search(trainer: Trainer, n_trials: int = 5) -> Optional[str]:
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        n_trials=n_trials,
    )
    logger.info("Best trial: %s", best_trial)
    return best_trial.checkpoint if best_trial and hasattr(best_trial, "checkpoint") else None

# -----------------------------------------------------------------------------
# Main Training Pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    # Clear Transformers cache directory (if it exists)
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        try:
            rmtree(cache_dir)
            logger.info("Transformers cache cleared.")
        except Exception as e:
            logger.warning(f"Failed to clear Transformers cache: {e}")
    else:
        logger.info("Transformers cache directory not found.")

    # Load emails from MongoDB.
    try:
        emails = load_emails_from_mongo()
    except Exception:
        logger.error("Failed to load emails from MongoDB. Exiting.")
        return

    data_dict = process_emails(emails)
    dataset = Dataset.from_dict(data_dict)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    raw_eval_dataset = Dataset.from_dict(eval_dataset.to_dict())
    logger.info("Dataset partitioned. Train columns: %s",
                train_dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=128)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=128)
    logger.info("After tokenization, train dataset columns: %s",
                train_dataset.column_names)

    train_dataset.set_format(type="torch", columns=[
                             "input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=[
                            "input_ids", "attention_mask", "label"])

    # MODEL INITIALIZATION: Load an existing model or create a new one.
    model_dir = "email_classifier_llm_latest"
    choice = input(
        f"Load an existing model from '{model_dir}'? (y/n): ").strip().lower()
    if choice == "y" and os.path.exists(model_dir):
        logger.info("Loading existing model from: %s", model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=NUM_LABELS)
    else:
        if choice == "y":
            logger.warning(
                "No model found at '%s'. Creating a new model.", model_dir)
        else:
            logger.info("Creating a new model.")
        model = model_init()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="output_llm",
        # Using eval_strategy instead of evaluation_strategy.
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        seed=RANDOM_SEED,
        load_best_model_at_end=True,
    )

    # Initialize Trainer with model_init so hyperparameter search can create new instances.
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Optional hyperparameter search.
    best_checkpoint = run_hyperparameter_search(trainer, n_trials=3)
    if best_checkpoint:
        logger.info("Resuming training from best checkpoint: %s",
                    best_checkpoint)
        trainer.train(resume_from_checkpoint=best_checkpoint)
    else:
        logger.info("Starting training from scratch.")
        trainer.train()

    logger.info("Model training complete.")
    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)
    print("Evaluation results:", eval_results)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info("Model and tokenizer saved to '%s'.", model_dir)

    # Post-training analysis of misclassifications.
    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions
    predicted_labels = np.argmax(logits, axis=-1)
    true_labels = raw_eval_dataset["label"]
    texts = raw_eval_dataset["text"]

    misclassified = []
    for i in range(len(true_labels)):
        if predicted_labels[i] != true_labels[i]:
            reason = heuristic_reason(
                texts[i], true_labels[i], predicted_labels[i])
            misclassified.append({
                "text_snippet": texts[i][:200],
                "actual": get_label_name(true_labels[i]),
                "predicted": get_label_name(predicted_labels[i]),
                "reason": reason,
            })

    accuracy = accuracy_score(true_labels, predicted_labels)
    logger.info("Post-training Accuracy on evaluation set: %.2f%%",
                accuracy * 100)
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

    if misclassified:
        logger.info("Misclassified examples:")
        for idx, mis in enumerate(misclassified, start=1):
            print(f"--- Misclassified Example {idx} ---")
            print(f"Text Snippet: {mis['text_snippet']}")
            print(
                f"Actual Label: {mis['actual']}, Predicted Label: {mis['predicted']}")
            print(f"Heuristic Reason: {mis['reason']}\n")
    else:
        logger.info("No misclassifications found on evaluation data.")


if __name__ == "__main__":
    main()
