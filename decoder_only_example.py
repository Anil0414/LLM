from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input prompt
prompt = "Once upon a time, there was a clever fox who"

# Encode the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text (decoder-only causal generation)
output = model.generate(
    inputs["input_ids"],
    max_length=50,       # total length including prompt
    do_sample=True,      # enable sampling for variety
    top_k=50,            # consider top 50 tokens at each step
    top_p=0.95,          # nucleus sampling
    temperature=0.7      # randomness factor
)

# Decode generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)