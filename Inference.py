from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = "./financial_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(sentences, model, tokenizer):
    """
    Predict sentiment for one or multiple sentences using a fine-tuned DistilBERT model.
    
    Args:
        sentences (str or list): A single sentence or a list of sentences to classify.
        model (torch.nn.Module): The trained model for sentiment classification.
        tokenizer (AutoTokenizer): The tokenizer used to preprocess input text.
    
    Returns:
        list: Predicted sentiment labels.
    """

    if isinstance(sentences, str):
        sentences = [sentences]  # Convert to list if a single sentence is provided

    # Tokenize input sentences (efficient batch processing)
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input tensors to GPU if available

    # Perform inference
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradients for efficiency
        outputs = model(**inputs)

    # Get predicted class (sentiment) from logits
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()  # Move back to CPU for processing

    # Map predictions to sentiment labels
    label_map = {0: "Negative Sentiment", 1: "Neutral Sentiment", 2: "Positive Sentiment"}
    return [label_map[pred] for pred in predictions]

# Test Example
test_sentences = [
    "The company's performance exceeded expectations.",
    "There are some risks, but overall it's stable.",
    "The stock crashed significantly today!"
]

predicted_sentiments = predict_sentiment(test_sentences, model, tokenizer)
for sentence, sentiment in zip(test_sentences, predicted_sentiments):
    print(f"Sentence: {sentence}\nPredicted Sentiment: {sentiment}\n")
