from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
prompt_text = "Really?"
input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

# Generate 100 tokens of text
output = model.generate(input_ids, max_length=100, num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
