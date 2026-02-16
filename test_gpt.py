from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load GPT-style causal model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input text (incomplete sentence)
text = "I forgot my password, how can I"

inputs = tokenizer(text, return_tensors="pt")

# Generate continuation
outputs = model.generate(
    **inputs,
    max_length=20,
    do_sample=True,
    temperature=0.7
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)