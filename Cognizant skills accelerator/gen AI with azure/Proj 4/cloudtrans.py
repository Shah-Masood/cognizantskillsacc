from datasets import load_dataset

# Load IMDB dataset (pre-labeled for sentiment analysis)
dataset = load_dataset("imdb")

# Split into training and testing sets
train_dataset = dataset["train"]
test_dataset = dataset["test"]

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load pre-trained DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

import numpy as np
from datasets import load_metric

# Load accuracy metric
metric = load_metric("accuracy")

# Evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Evaluate
results = trainer.evaluate()
print("Evaluation Results:", results)

from azureml.core import Workspace, Model

# Connect to your Azure ML workspace
ws = Workspace.from_config()

# Register the trained model
model = Model.register(
    workspace=ws,
    model_name="sentiment-analysis-transformer",
    model_path="./results",  # Path to saved model
    description="Fine-tuned DistilBERT model for sentiment analysis",
)
print(f"Model registered: {model.name} (Version: {model.version})")

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model_path = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Inference function
def run(data):
    inputs = tokenizer(data["text"], truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).tolist()
    return {"prediction": predictions}
