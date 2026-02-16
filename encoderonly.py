from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT tokenizer and model for classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Input sentence
sentence = "I absolutely love this movie!"

# Tokenize input
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# Forward pass (get logits)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to predicted class
predicted_class = torch.argmax(logits, dim=1).item()
labels = {0: "Negative", 1: "Positive"}

print(f"Sentence: '{sentence}'")
print(f"Predicted sentiment: {labels[predicted_class]}")