# Call our local model to generate text completions
#
# Thomas Lundqvist, 2026, use freely!

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = "./scifi-gpt2"   # our techno sci-fi speaker

#model_dir = "gpt2"   # original model

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

def complete(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    print()
    print("Prompt:", prompt)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# try one sci-fi prompt and one unrelated, will the unrelated also get
# scifi output?

complete("Quantum space")
complete("I like to eat")
complete("I want to")
complete("Kan du svenska?")
print()
