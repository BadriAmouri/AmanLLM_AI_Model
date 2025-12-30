# Darija QA Pipeline Guide

## 🎯 Overview

This pipeline allows users to ask questions in **Darija (Moroccan Arabic)** and receive answers in MSA (Modern Standard Arabic).

### Pipeline Flow:
```
User Question (Darija) 
    ↓
Translation Model (Darija → MSA)
    ↓
QA Model (Fine-tuned Qwen 2.5-3B with LoRA)
    ↓
Answer (MSA)
```

## 📁 Project Structure

```
FinalModelPipeline/
├── main.py                    # 🚀 Main pipeline (RUN THIS)
├── translation_model.py       # Translation model standalone
├── qa_model.py               # QA model standalone
├── requirements.txt          # Python dependencies
├── translated_model/         # Translation model files
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
└── qa_model/                 # QA model LoRA adapter
    ├── adapter_config.json
    ├── adapter_model.safetensors (228MB)
    └── tokenizer files...
```

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `transformers` - Hugging Face models
- `peft` - LoRA adapter loading
- `torch` - Deep learning framework

### 2. Verify Model Files

Ensure you have:
- ✅ `translated_model/` folder with translation model
- ✅ `qa_model/` folder with LoRA adapter

### 3. First Run Note ⚠️

**The first time you run `main.py`, it will download the base QA model (~6GB) from Hugging Face.**

This is a ONE-TIME download. The model will be cached at:
- Mac/Linux: `~/.cache/huggingface/`
- Windows: `C:\Users\<user>\.cache\huggingface\`

Subsequent runs will use the cached model.

## 🚀 Usage

### Run the Pipeline

```bash
python main.py
```

### Example Session

```
🚀 DARIJA QA PIPELINE
══════════════════════════════════════════════════════════════
Device: mps

[1/2] Loading Translation Model (Darija ↔ MSA)...
✓ Translation model loaded successfully

[2/2] Loading QA Model (with LoRA adapter)...
  → Loading base model (this may download ~6GB on first run)...
  → Loading LoRA adapter...
✓ QA model loaded successfully

══════════════════════════════════════════════════════════════
✅ PIPELINE READY!
══════════════════════════════════════════════════════════════

🧪 Running test example...

──────────────────────────────────────────────────────────────
📝 Query (Darija): شنو هي الذكاء الاصطناعي؟
──────────────────────────────────────────────────────────────
🔄 Translation (MSA): ما هو الذكاء الاصطناعي؟
──────────────────────────────────────────────────────────────
💡 Answer: الذكاء الاصطناعي هو فرع من علوم الحاسوب...
──────────────────────────────────────────────────────────────

💬 INTERACTIVE MODE
══════════════════════════════════════════════════════════════
Enter your questions in Darija. Type 'quit' or 'exit' to stop.

🗣️  Your question (Darija): 
```

## 📊 Models Information

### Translation Model
- **Type:** Sequence-to-Sequence (T5/mBART-based)
- **Task:** Darija ↔ MSA translation
- **Size:** ~1GB
- **Location:** `./translated_model/`

### QA Model
- **Base Model:** Qwen/Qwen2.5-3B-Instruct
- **Fine-tuning:** LoRA (Low-Rank Adaptation)
- **Adapter Size:** 228MB
- **Base Model Size:** ~6GB (downloaded on first run)
- **Location:** `./qa_model/` (adapter only)

### LoRA Configuration
```json
{
  "lora_rank": 32,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ]
}
```

## 🛠️ Troubleshooting

### Issue: Downloading 6GB every time

**Problem:** Base model re-downloads on each run.

**Solution:** The code now uses Hugging Face cache. The download happens only once.

If you want to avoid internet dependency:
1. Run once to download
2. Find cache location: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct`
3. Update `QA_BASE_MODEL` in `main.py` to this local path

### Issue: Out of memory

**Solutions:**
- Use CPU instead of GPU/MPS (slower but uses less RAM)
- Reduce `max_new_tokens` in the QA function
- Close other applications

### Issue: Translation model not found

**Error:** `FileNotFoundError: Translation model not found at ./translated_model`

**Solution:** Ensure the `translated_model` folder exists with all files.

### Issue: Slow on first run

This is normal! The pipeline loads two models:
1. Translation model (~1GB)
2. QA base model + adapter (~6GB on first run)

After first run, loading is much faster due to caching.

## 🔍 Advanced Usage

### Using the Models Separately

**Translation only:**
```bash
python translation_model.py
```

**QA only:**
```bash
python qa_model.py
```

### Customizing Generation Parameters

Edit `main.py` in the `answer_question` function:

```python
output_ids = qa_model.generate(
    **inputs,
    max_new_tokens=256,      # Longer answers
    temperature=0.7,         # Creativity (0.1-1.0)
    do_sample=True,          # Enable sampling
    top_p=0.9,              # Nucleus sampling
    repetition_penalty=1.1   # Avoid repetition
)
```

### Batch Processing

You can modify the pipeline to process multiple questions:

```python
questions = [
    "شنو هو الذكاء الاصطناعي؟",
    "كيفاش نتعلم البرمجة؟",
    "واش الماء مهم للصحة؟"
]

for q in questions:
    result = process_darija_query(q)
    print(result)
```

## 📝 Pipeline Functions

### `translate_darija_to_msa(darija_text)`
Translates Darija text to MSA.

**Parameters:**
- `darija_text` (str): Input text in Darija

**Returns:**
- str: Translated text in MSA

### `answer_question(msa_question, max_new_tokens=256)`
Answers a question using the QA model.

**Parameters:**
- `msa_question` (str): Question in MSA
- `max_new_tokens` (int): Maximum length of answer

**Returns:**
- str: Generated answer

### `process_darija_query(darija_query, show_steps=True)`
Complete pipeline from Darija query to answer.

**Parameters:**
- `darija_query` (str): Question in Darija
- `show_steps` (bool): Print intermediate steps

**Returns:**
- dict: Contains darija_query, msa_translation, and answer

## 🎓 Example Queries

Try these example questions in Darija:

```
شنو هي البرمجة؟
(What is programming?)

كيفاش نحسن الذاكرة ديالي؟
(How can I improve my memory?)

واش القهوة مضرة للصحة؟
(Is coffee bad for health?)

شنو هما فوائد الرياضة؟
(What are the benefits of exercise?)

كيفاش نتعلم لغة جديدة؟
(How do I learn a new language?)
```

## 💾 Memory Requirements

- **Minimum RAM:** 8GB
- **Recommended RAM:** 16GB
- **GPU/MPS VRAM:** 6GB+ (if using GPU acceleration)

## 🚀 Performance Tips

1. **Use GPU/MPS** if available (automatically detected)
2. **Keep models loaded** - Don't restart between queries
3. **Use batch processing** for multiple queries
4. **Adjust max_new_tokens** based on answer length needs

## 📄 License

This project uses:
- Qwen 2.5 (Alibaba Cloud) - Check Qwen license
- Your fine-tuned models - Your terms

## 🤝 Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are complete
4. Check device (CPU/GPU/MPS) compatibility

---

**Happy Questioning! 🎉**
