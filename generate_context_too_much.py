from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = "gpt2"   # original model

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

def complete(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    except IndexError:
        n_input = inputs["input_ids"].shape[1]
        print(f"Context window exceeded: {n_input} input tokens + 50 max_new_tokens = {n_input + 50} > 1024 limit")
        return
    print()
    print("Prompt:", prompt)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# exceed context length (1025 tokens > GPT-2's 1024 limit)
# 95 copies should fit, 99 should be too much
prompt = "hello world, this sentence will make 10 tokens!" * 95
count = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
print("Token count:", count, "Total count with new tokens:", count + 50)
complete(prompt)


