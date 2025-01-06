from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the dataset


dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")  # Load financial sentiment dataset
print(dataset[100])  # Print the 100th sample to inspect the data
print(dataset.column_names)  # Display the column names; typically 'sentence' and 'label'

# Step 2: Split the dataset into training and evaluation sets
# The dataset is split into 80% for training and 20% for evaluation (test) using a random seed to ensure reproducibility.
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)  # Split the data
train_dataset = split_dataset["train"]  # Training dataset
eval_dataset = split_dataset["test"]  # Evaluation (test) dataset

# Step 3: Load the pre-trained tokenizer and model
# Tokenizer: Converts raw text (sentences) into token IDs that the model understands.
# Model: We are using "distilbert-base-uncased", a lighter version of BERT optimized for faster performance.

model_name = "distilbert-base-uncased"  # Model name for a lightweight BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load the tokenizer for text conversion
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Load the model for 3 sentiment labels

# Step 4: Preprocess the dataset by tokenizing the sentences
# The tokenizer converts each sentence into a series of token IDs (numbers that represent the words).
# Padding ensures that all sentences are of the same length, truncation limits the sentence to a fixed length.
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")  # Tokenization and padding

# Apply preprocessing to the entire training and evaluation datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)  # Tokenize training data
eval_dataset = eval_dataset.map(preprocess_function, batched=True)  # Tokenize evaluation data

# Step 5: Format the datasets for PyTorch compatibility
# The model requires the dataset to be in a specific format: it expects 'input_ids', 'attention_mask', and 'label' fields.
# The 'input_ids' are the token IDs, and 'attention_mask' indicates which tokens are real (1) or padding (0).
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  # Set PyTorch format for training
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  # Set PyTorch format for evaluation

# Step 6: Define the training arguments
# These arguments control the training process, such as batch size, learning rate, number of epochs, and more.
training_args = TrainingArguments(
    output_dir="./results",  # Directory where model and checkpoints will be saved
    eval_strategy="epoch",  # Evaluate the model at the end of each epoch
    save_strategy="epoch",  # Save the model at the end of each epoch
    logging_dir="./logs",  # Directory to save logs
    learning_rate=2e-5,  # Learning rate used for optimization
    per_device_train_batch_size=16,  # Batch size used during training (how many samples per batch)
    num_train_epochs=3,  # Number of times the model will train on the entire dataset
    weight_decay=0.01,  # Regularization parameter to prevent overfitting
    report_to="none",  # Disable reporting to external tools like WandB
)

# Step 7: Initialize the Trainer
# The Trainer handles the training and evaluation of the model using the specified arguments and datasets.
trainer = Trainer(
    model=model,  # Pre-trained model to fine-tune
    args=training_args,  # Training arguments that control the training process
    train_dataset=train_dataset,  # Tokenized training dataset
    eval_dataset=eval_dataset,  # Tokenized evaluation dataset
)

# Step 8: Fine-tune the model
# The `train()` function starts the fine-tuning process where the model adjusts its weights based on the training data.
trainer.train()

# Step 9: Save the fine-tuned model for future use
# After training, the model and tokenizer are saved to disk for later use (inference or further training).
model.save_pretrained("./financial_sentiment_model")  # Save the fine-tuned model
tokenizer.save_pretrained("./financial_sentiment_model")  # Save the tokenizer
