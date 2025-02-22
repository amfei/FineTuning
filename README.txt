Financial Sentiment Analysis with DistilBERT Overview
This project fine-tunes a DistilBERT model on financial sentiment analysis using the Financial PhraseBank dataset. The dataset consists of financial news and reports labeled as positive, negative, or neutral.


Train:


 Features
Uses DistilBERT, a lightweight variant of BERT for efficiency.
Prepares and tokenizes data from the Financial PhraseBank dataset.
Fine-tunes the model on financial sentiment classification.
Uses Trainer API for efficient training and evaluation.
Supports mixed-precision training (fp16) for faster performance on GPUs.
Saves the best-performing model automatically.




Inference 
This script loads a fine-tuned DistilBERT model for financial sentiment classification and predicts the sentiment (Positive, Neutral, or Negative) of input sentences. 
It supports batch processing and GPU acceleration for efficient inference.

Features
Supports both single sentence and batch processing
Automatically detects GPU availability for fast inference
Efficient tokenization for optimized performance
Handles exceptions and ensures robustness

Installation

Before running the script, install the necessary dependencies:

pip install transformers torch

Model & Tokenizer

This script assumes the fine-tuned model and tokenizer are stored in ./financial_sentiment_model.
If you haven't trained the model yet, refer to the training script to fine-tune it first.

