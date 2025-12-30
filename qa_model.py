# qa_model_test.py (or .ipynb)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from jinja2 import Environment, FileSystemLoader
import os

# =========================
# Config
# =========================
MODEL_DIR = "./qa_model"  # folder with config.json, adapter, tokenizer, template
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# =========================
# Load base model
# =========================
# If your final model already has LoRA merged, you can load directly:
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
model.to(DEVICE)
print(f"[✓] Model loaded on {DEVICE}")

# =========================
# Load chat template
# =========================
env = Environment(loader=FileSystemLoader(MODEL_DIR))
template = env.get_template("chat_template.jinja")

def format_prompt(question: str):
    """Render the chat template with the user question"""
    return template.render(instruction=question)

# =========================
# QA function
# =========================
def answer_question(question: str, max_new_tokens=512):
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# =========================
# Interactive QA loop
# =========================
print("Enter 'quit' to exit")
while True:
    q = input("\nQuestion: ")
    if q.lower() == "quit":
        break
    answer = answer_question(q)
    print(f"Answer: {answer}")
