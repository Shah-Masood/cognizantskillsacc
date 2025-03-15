# Step 1: Import Necessary Libraries
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
)
from sklearn.metrics import classification_report

# Step 2: Load and Prepare Dataset
dataset = load_dataset("imdb")

# Select a subset of 200 training samples and 50 test samples for quick training
train_data = dataset["train"].shuffle(seed=42).select(range(200))
test_data = dataset["test"].shuffle(seed=42).select(range(50))

# Re-label data (IMDB has binary labels; we introduce a neutral class)
def relabel_data(example):
    text = example["text"]
    sentiment = example["label"]  # IMDB: 0 = Negative, 1 = Positive
    if sentiment == 1:
        return {"text": text, "label": 2}  
    return {"text": text, "label": sentiment}  

train_data = train_data.map(relabel_data)
test_data = test_data.map(relabel_data)

# Step 3: Load Pretrained Model & Tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Tokenize Dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save Fine-Tuned Model
model.save_pretrained("./fine_tuned_sentiment")
tokenizer.save_pretrained("./fine_tuned_sentiment")

# Step 9: Evaluate Model Performance
results = trainer.evaluate()
print("Evaluation Results:", results)

# Step 10: Detailed Performance Metrics
predictions = trainer.predict(tokenized_test)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = tokenized_test["label"]

# Print Classification Report
report = classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"])
print(report)
