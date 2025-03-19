import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 2: Load and Preprocess the IMDB Dataset
dataset = load_dataset("imdb")

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Split dataset into training and testing
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Step 3: Load Model and Define Training Arguments
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./distilbert-imdb",  # Save directory
    evaluation_strategy="epoch",  # Evaluate every epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Step 4: Train the Model
trainer.train()

# Step 5: Save the Fine-Tuned Model
model.save_pretrained("./distilbert-imdb")
tokenizer.save_pretrained("./distilbert-imdb")

# Step 6: Evaluate the Model
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

results = trainer.evaluate()
accuracy = compute_metrics((results["eval_loss"], test_dataset["labels"]))
print(f"Test Accuracy: {accuracy['accuracy']:.4f}")