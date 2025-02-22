from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

# Step 1: Load the dataset
# - The dataset used is "financial_phrasebank", which contains financial sentiment analysis data.
# - "sentences_allagree" means that all annotators agreed on the sentiment labels.
dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")

# Step 2: Split into training (80%) and evaluation (20%) sets
# - The data is shuffled before splitting to ensure a more generalized model.
split_dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

# Step 3: Load tokenizer and model
# - Using DistilBERT (a lighter version of BERT) for sequence classification.
# - Tokenizer converts text into numerical representations (token IDs).
# - The model is initialized with 3 output labels (positive, neutral, negative sentiment).
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Step 4: Tokenize dataset
# - `truncation=True`: Ensures sentences longer than max length are truncated.
# - `padding=True`: Pads shorter sentences so all inputs have equal length.
# - `max_length=128`: Limits the number of tokens to prevent excessive computation.
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True, max_length=128)

# Tokenizing the dataset in batches for better efficiency
train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=1000)
eval_dataset = eval_dataset.map(preprocess_function, batched=True, batch_size=1000)

# Step 5: Convert datasets to PyTorch tensors
# - `input_ids`: The tokenized input sequences.
# - `attention_mask`: Indicates which tokens are actual words (1) and which are padding (0).
# - `label`: The sentiment label of the sentence.
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Step 6: Use DataCollatorWithPadding
# - This ensures dynamic padding during training, optimizing GPU memory usage.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Where the model checkpoints will be saved
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model checkpoints at the end of each epoch
    logging_dir="./logs",  # Directory for logs
    learning_rate=2e-5,  # Optimized learning rate for fine-tuning
    per_device_train_batch_size=16,  # Number of samples per training batch
    per_device_eval_batch_size=16,  # Number of samples per evaluation batch
    num_train_epochs=3,  # Number of times the model will see the entire dataset
    weight_decay=0.01,  # Regularization to prevent overfitting
    report_to="none",  # Disable external logging (e.g., WandB)
    load_best_model_at_end=True,  # Save the best-performing model after training
    fp16=True,  # Use mixed-precision training for faster performance on GPUs
)

# Step 8: Initialize Trainer
# - `Trainer` is a high-level class that simplifies fine-tuning Hugging Face models.
trainer = Trainer(
    model=model,  # The pre-trained model to fine-tune
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # The processed training dataset
    eval_dataset=eval_dataset,  # The processed evaluation dataset
    data_collator=data_collator,  # Handles dynamic padding
)

# Step 9: Train the model
trainer.train()

# Step 10: Evaluate the model after training
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")  # Displays model performance

# Step 11: Save the fine-tuned model and tokenizer
# - Saving ensures the model can be reloaded for future use without retraining.
model.save_pretrained("./financial_sentiment_model")
tokenizer.save_pretrained("./financial_sentiment_model")
