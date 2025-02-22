# The dataset used here is the "financial_phrasebank", which contains sentences with sentiment labels.
# Sentiment Labels: 
# 0 = Negative Sentiment
# 1 = Neutral Sentiment
# 2 = Positive Sentiment

README: Financial Sentiment Analysis with DistilBERT Overview
This project fine-tunes a DistilBERT model on financial sentiment analysis using the Financial PhraseBank dataset. The dataset consists of financial news and reports labeled as positive, negative, or neutral.

 Features
Uses DistilBERT, a lightweight variant of BERT for efficiency.
Prepares and tokenizes data from the Financial PhraseBank dataset.
Fine-tunes the model on financial sentiment classification.
Uses Trainer API for efficient training and evaluation.
Supports mixed-precision training (fp16) for faster performance on GPUs.
Saves the best-performing model automatically.

