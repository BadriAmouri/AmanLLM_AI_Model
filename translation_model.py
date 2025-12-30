from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# ===============================
# Local model path
# ===============================
MODEL_DIR = "./translated_model"  # folder containing model.safetensors, config.json, tokenizer files

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model folder not found at {MODEL_DIR}")

# ===============================
# Load model
# ===============================
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# ===============================
# Device selection (Mac safe)
# ===============================
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(device)
model.eval()

print(f"Model loaded successfully on {device}!")

# ===============================
# Translation function
# ===============================
def translate(text, direction="darija_to_msa"):
    prefix = "translate Darija to MSA: " if direction == "darija_to_msa" else "translate MSA to Darija: "
    input_text = prefix + text

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===============================
# Test
# ===============================
print("\n" + "=" * 60)
print("MODEL READY FOR TESTING!")
print("=" * 60)

test_text = "ولدي راهو سخون بزاف"
result = translate(test_text, "darija_to_msa")
print("\nTest Translation:")
print(f"Darija → MSA: {test_text}")
print(f"Result: {result}")

# ===============================
# Interactive mode
# ===============================
print("\nEnter 'quit' to exit")

while True:
    direction = input("\nDirection (1=Darija→MSA, 2=MSA→Darija): ")
    if direction.lower() == "quit":
        break

    text = input("Enter text: ")
    if text.lower() == "quit":
        break

    dir_map = {"1": "darija_to_msa", "2": "msa_to_darija"}
    result = translate(text, dir_map.get(direction, "darija_to_msa"))
    print(f"Translation: {result}")
