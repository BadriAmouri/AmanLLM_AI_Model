"""
Darija QA Pipeline
==================
This pipeline takes a Darija query, translates it to MSA, 
then answers it using the fine-tuned QA model.

Pipeline: Darija Query → MSA Translation → QA Model → Answer
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# ===============================
# Configuration
# ===============================
TRANSLATION_MODEL_DIR = "./translated_model"
QA_MODEL_DIR = "./qa_model"
QA_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # Base model for QA

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"\n{'='*70}")
print(f"🚀 DARIJA QA PIPELINE")
print(f"{'='*70}")
print(f"Device: {DEVICE}")

# ===============================
# Load Translation Model
# ===============================
print("\n[1/2] Loading Translation Model (Darija ↔ MSA)...")

if not os.path.exists(TRANSLATION_MODEL_DIR):
    raise FileNotFoundError(f"Translation model not found at {TRANSLATION_MODEL_DIR}")

translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_DIR)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_DIR)
translation_model.to(DEVICE)
translation_model.eval()

print(f"✓ Translation model loaded successfully")

# ===============================
# Load QA Model
# ===============================
print("\n[2/2] Loading QA Model (with LoRA adapter)...")

if not os.path.exists(QA_MODEL_DIR):
    raise FileNotFoundError(f"QA model not found at {QA_MODEL_DIR}")

qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_DIR, trust_remote_code=True)

# Load base model
print("  → Loading base model (this may download ~6GB on first run)...")
qa_base_model = AutoModelForCausalLM.from_pretrained(
    QA_BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16  # Use fp16 for efficiency
)

# Load LoRA adapter
print("  → Loading LoRA adapter...")
qa_model = PeftModel.from_pretrained(qa_base_model, QA_MODEL_DIR, device_map="auto")
qa_model.to(DEVICE)
qa_model.eval()

print(f"✓ QA model loaded successfully")

# ===============================
# Pipeline Functions
# ===============================

def translate_darija_to_msa(darija_text):
    """Translate Darija text to Modern Standard Arabic"""
    input_text = "translate Darija to MSA: " + darija_text
    inputs = translation_tokenizer(input_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output_ids = translation_model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    return translation_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def answer_question(msa_question, max_new_tokens=256):
    """Answer a question in MSA using the QA model"""
    # Format using chat template
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": msa_question}
    ]
    
    # Apply chat template
    prompt = qa_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = qa_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output_ids = qa_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # Decode and extract answer
    full_response = qa_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in full_response:
        answer = full_response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        return answer
    
    return full_response


def process_darija_query(darija_query, show_steps=True):
    """
    Complete pipeline: Darija Query → MSA Translation → QA Answer
    """
    if show_steps:
        print(f"\n{'─'*70}")
        print(f"📝 Query (Darija): {darija_query}")
        print(f"{'─'*70}")
    
    # Step 1: Translate to MSA
    msa_translation = translate_darija_to_msa(darija_query)
    if show_steps:
        print(f"🔄 Translation (MSA): {msa_translation}")
        print(f"{'─'*70}")
    
    # Step 2: Get answer from QA model
    answer = answer_question(msa_translation)
    if show_steps:
        print(f"💡 Answer: {answer}")
        print(f"{'─'*70}\n")
    
    return {
        "darija_query": darija_query,
        "msa_translation": msa_translation,
        "answer": answer
    }

# ===============================
# Interactive Mode
# ===============================

print(f"\n{'='*70}")
print(f"✅ PIPELINE READY!")
print(f"{'='*70}")

# Test example
print("\n🧪 Running test example...")
test_query = "شنو هي الذكاء الاصطناعي؟"
process_darija_query(test_query)

# Interactive loop
print("\n" + "="*70)
print("💬 INTERACTIVE MODE")
print("="*70)
print("Enter your questions in Darija. Type 'quit' or 'exit' to stop.\n")

while True:
    user_input = input("🗣️  Your question (Darija): ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q', '']:
        print("\n👋 Goodbye!")
        break
    
    try:
        process_darija_query(user_input)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
