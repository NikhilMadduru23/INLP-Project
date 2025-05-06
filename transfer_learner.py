import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import evaluate
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
import wandb
import json

# Initialize WandB
wandb.init(project="inlp_project", name="distilbert-finetuning-atis", config={
    "model": "distilbert-base-uncased",
    "dataset": "atis",
    "num_epochs": 3,
    "batch_size": 16
})

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset("csv", data_files={
    'train': "./data/processed/atis_train.csv", 
    'test': "./data/processed/atis_test.csv"
})
print(f"Original train size: {len(dataset['train'])}, test size: {len(dataset['test'])}")

# Subsample the dataset
def subsample_dataset(dataset_split, fraction=0.1):
    dataset_split = dataset_split.shuffle(seed=42)
    num_samples = int(len(dataset_split) * fraction)
    print(f"Subsampling {len(dataset_split)} to {num_samples} samples")
    return dataset_split.select(range(num_samples))

reduced_train = subsample_dataset(dataset["train"], fraction=1)
reduced_test = subsample_dataset(dataset["test"], fraction=1)
reduced_dataset = {"train": reduced_train, "test": reduced_test}
print(f"Reduced train size: {len(reduced_train)}, reduced test size: {len(reduced_test)}")

# Number of unique intents
num_intents = len(set(reduced_dataset['train']['encoded_intent']))
print(f"Number of intents: {num_intents}")

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_intents)
model.to(device)

# Tokenize dataset
def tokenize_function(examples):
    texts = examples["text"]
    valid_texts = [t if isinstance(t, str) and t.strip() else "DUMMY" for t in texts]
    tokenized = tokenizer(valid_texts, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = examples["encoded_intent"]
    return tokenized

# Apply tokenization to each split
try:
    tokenized_train = reduced_dataset["train"].map(tokenize_function, batched=True)
    tokenized_test = reduced_dataset["test"].map(tokenize_function, batched=True)
    tokenized_dataset = {"train": tokenized_train, "test": tokenized_test}

    for split in ["train", "test"]:
        tokenized_dataset[split] = tokenized_dataset[split].remove_columns(["text", "encoded_intent"])
        tokenized_dataset[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        metrics = {
            "accuracy": accuracy["accuracy"],
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        return metrics

    # Custom callback to collect loss
    class LossLoggingCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs.get("loss") is not None:
                self.train_losses.append(logs["loss"])
                wandb.log({"train/loss": logs["loss"], "step": state.global_step})
            if logs.get("eval_loss") is not None:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
                wandb.log({"eval/loss": logs["eval_loss"], "step": state.global_step})

    loss_callback = LossLoggingCallback()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,
        save_strategy="epoch",
        report_to="wandb"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[loss_callback]
    )
    
    # Calculate and print expected steps
    steps_per_epoch = len(tokenized_dataset["train"]) / training_args.per_device_train_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    print(f"Expected steps per epoch: {steps_per_epoch:.0f}, Total steps: {total_steps:.0f}")
    
    # Train the model
    print("\nTraining the model...")
    trainer.train()

    # Evaluate trained model
    print("\nEvaluating trained model...")
    trained_metrics = trainer.evaluate(tokenized_dataset["test"])
    print("Trained model metrics:", trained_metrics)
    wandb.log({"trained/accuracy": trained_metrics["eval_accuracy"],
               "trained/precision": trained_metrics["eval_precision"],
               "trained/recall": trained_metrics["eval_recall"],
               "trained/f1": trained_metrics["eval_f1"],
               "trained/loss": trained_metrics["eval_loss"]})

    # Save metrics to JSON file
    metrics_report = {
        "trained_metrics": trained_metrics
    }
    try:
        with open("./metrics/metrics_report.json", "w") as f:
            json.dump(metrics_report, f, indent=4)
        print("Metrics report saved to ./metrics/metrics_report.json")
    except Exception as e:
        print(f"Failed to save metrics report to JSON: {e}")

    # Plot and save training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_callback.train_losses)), loss_callback.train_losses, label="Training Loss", color='blue')
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/training_loss_plot.png")
    plt.close()

    # Plot and save evaluation loss
    if loss_callback.eval_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_callback.eval_steps, loss_callback.eval_losses, label="Evaluation Loss", marker='o', color='orange')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Evaluation Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("./plots/eval_loss_plot.png")
        plt.close()

    # Plot and save combined loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_callback.train_losses)), loss_callback.train_losses, label="Training Loss", color='blue')
    if loss_callback.eval_losses:
        plt.plot(loss_callback.eval_steps, loss_callback.eval_losses, label="Evaluation Loss", marker='o', color='orange')
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/loss_plot.png")
    plt.close()

    # Log all plots to WandB
    wandb.log({
        "training_loss_plot": wandb.Image("./plots/training_loss_plot.png"),
        "eval_loss_plot": wandb.Image("./plots/eval_loss_plot.png") if loss_callback.eval_losses else None,
        "combined_loss_plot": wandb.Image("./plots/loss_plot.png")
    })

    # Save model and tokenizer
    trainer.save_model("./saved_models/intent_classifier")
    tokenizer.save_pretrained("./saved_models/intent_classifier")
    print("Model and tokenizer saved to ./saved_models/intent_classifier")

    # Finish WandB run
    wandb.finish()

except ValueError as e:
    print(f"Tokenization failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")