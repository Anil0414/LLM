from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Example sentence with a masked token
sentence = "The cat sat on the [MASK]."

# Tokenize input
inputs = tokenizer(sentence, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Get the predicted token for [MASK]
masked_index = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
predicted_index = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_index)

print(f"Predicted word: {predicted_token}")