# Fine tuning example
#
# Thomas Lundqvist, 2026, use freely!

from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load base model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Create a small training dataset (technical sci-fi sentences)
train_text = """
The quantum drive is destabilizing — reroute power through the secondary manifold.
Biosigns on deck seven are fluctuating beyond acceptable variance thresholds.
Initiating emergency cryostasis protocol, all non-essential crew to hibernation pods.
The wormhole aperture collapsed before we could complete the transit sequence.
Gravitational shear from the neutron star is warping our sensor array.
Subspace interference is degrading the ansible link to command.
Hull integrity at sixty-three percent — ablative plating on sectors four through nine is failing.
The AI core has entered a recursive diagnostic loop and is unresponsive to manual override.
Antimatter containment field nominal, but the magnetic bottle is showing micro-fractures.
We are being scanned by an unknown vessel — energy signature matches no known civilization.
The terraforming nanobots have mutated beyond their original programming constraints.
Tachyon emissions from the artifact suggest it predates the formation of this star system.
Navigation systems are offline — we are computing our position by stellar parallax.
The pathogen is rewriting host DNA at the cellular level faster than our countermeasures can adapt.
Decompression in cargo bay three — emergency bulkheads have sealed the affected section.
The alien signal follows a base-12 mathematical structure we have only partially decoded.
Fusion reactor output is nominal but plasma conduit B7 is running forty degrees above tolerance.
Long-range sensors detect a debris field consistent with a destroyed colony ship.
The cybernetic interface is causing neural feedback — disengage the cortical implant immediately.
Orbital decay is accelerating — we have approximately ninety minutes before atmospheric entry.
""" * 20  # Repeat to give the trainer more data

def tokenize(element):
    return tokenizer(element["text"], truncation=True, max_length=64)

# 3. Tokenize the dataset
dataset = Dataset.from_dict({"text": [l for l in train_text.split("\n") if l.strip()]})
dataset = dataset.map(tokenize, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 uses causal LM, not masked LM
)

# 4. Set training arguments
training_args = TrainingArguments(
    output_dir="./scifi-gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="no",
    logging_steps=10,
    learning_rate=5e-5,
    report_to="none",
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# 6. Save the fine-tuned model
trainer.save_model("./scifi-gpt2")
tokenizer.save_pretrained("./scifi-gpt2")
