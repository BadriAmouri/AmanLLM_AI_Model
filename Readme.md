# Darija QA Pipeline Project

## 📋 Overview

This project implements a complete pipeline for answering questions in **Darija (Moroccan Arabic)**:
1. User asks a question in **Darija**
2. Translation model converts it to **MSA (Modern Standard Arabic)**
3. QA model (fine-tuned Qwen 2.5-3B with LoRA) generates an **answer**

---

## 🛠️ Environment Setup

### 1. Create Conda Environment

```bash
# Create the environment
conda env create -f environment.yml

# Activate it
conda activate darija-translation

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 2. Install Additional Requirements

```bash
pip install -r requirements.txt
```

**Key packages:**
- `transformers` - Hugging Face models
- `peft` - LoRA adapter loading
- `torch` - Deep learning framework

---

## 📁 Project Folder Structure

```
FinalModelPipeline/
│
├── main.py                          # 🚀 MAIN PIPELINE - Run this!
├── translation_model.py             # Translation model (standalone)
├── qa_model.py                      # QA model (standalone)
├── verify_setup.py                  # Setup verification script
│
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment file
├── Readme.md                        # This file
├── PIPELINE_GUIDE.md               # Detailed pipeline documentation
│
├── translated_model/                # 📦 Translation Model (Darija ↔ MSA)
│   ├── config.json                  # Model configuration
│   ├── model.safetensors           # Model weights (~1GB)
│   ├── tokenizer.json              # Tokenizer vocabulary
│   ├── tokenizer_config.json       # Tokenizer configuration
│   ├── special_tokens_map.json     # Special tokens mapping
│   └── [other tokenizer files]     # Additional tokenizer files
│
└── qa_model/                        # 📦 QA Model LoRA Adapter
    ├── adapter_config.json          # LoRA adapter configuration
    ├── adapter_model.safetensors   # LoRA weights (228MB)
    ├── config.json                  # Base model reference config
    ├── chat_template.jinja         # Chat template for QA
    ├── tokenizer.json              # Tokenizer vocabulary
    ├── tokenizer_config.json       # Tokenizer configuration
    ├── special_tokens_map.json     # Special tokens mapping
    ├── added_tokens.json           # Additional tokens
    ├── vocab.json                   # Vocabulary file
    ├── merges.txt                   # BPE merges
    └── [other tokenizer files]     # Additional tokenizer files

Note: The base QA model (Qwen/Qwen2.5-3B-Instruct ~6GB) will be
      downloaded automatically on first run to:
      ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/
```

---

## ⚙️ Required Model Files

### ✅ Translation Model (`translated_model/`)
- **Purpose:** Translate Darija ↔ MSA
- **Type:** Sequence-to-Sequence model
- **Size:** ~1GB
- **Required files:**
  - `model.safetensors` (model weights)
  - `config.json` (model config)
  - `tokenizer.json` (tokenizer)
  - `tokenizer_config.json`
  - `special_tokens_map.json`

### ✅ QA Model Adapter (`qa_model/`)
- **Purpose:** Answer questions in MSA
- **Type:** LoRA adapter for Qwen 2.5-3B
- **Size:** 228MB (adapter only)
- **Base Model:** Qwen/Qwen2.5-3B-Instruct (~6GB, auto-downloaded)
- **Required files:**
  - `adapter_model.safetensors` (LoRA weights)
  - `adapter_config.json` (LoRA config)
  - `config.json` (model config)
  - `chat_template.jinja` (chat template)
  - All tokenizer files

---

## 🚀 Usage

### Step 1: Verify Setup (Recommended)

```bash
python verify_setup.py
```

This checks if all required files and packages are present.

### Step 2: Run the Pipeline

```bash
python main.py
```

### Step 3: Ask Questions!

```
🗣️  Your question (Darija): شنو هي البرمجة؟
```

**Output:**
```
──────────────────────────────────────────────────────────────
📝 Query (Darija): شنو هي البرمجة؟
──────────────────────────────────────────────────────────────
🔄 Translation (MSA): ما هي البرمجة؟
──────────────────────────────────────────────────────────────
💡 Answer: البرمجة هي عملية كتابة التعليمات للحاسوب...
──────────────────────────────────────────────────────────────
```

---

## 📊 Model Details

### Translation Model
| Property | Value |
|----------|-------|
| Architecture | Seq2Seq (T5/mBART-based) |
| Task | Darija ↔ MSA Translation |
| Size | ~1GB |
| Location | `./translated_model/` |

### QA Model
| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 32 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Adapter Size | 228MB |
| Base Model Size | ~6GB (cached after first download) |
| Location | `./qa_model/` (adapter), `~/.cache/huggingface/` (base) |

---

## ⚠️ Important Notes

### First Run
- **The first run will download ~6GB** (Qwen base model from Hugging Face)
- This is a **ONE-TIME download**
- Subsequent runs use the cached model
- Be patient during first run!

### Memory Requirements
- **Minimum RAM:** 8GB
- **Recommended RAM:** 16GB
- **GPU/MPS VRAM:** 6GB+ (optional, for faster inference)

### Device Selection
The pipeline automatically detects and uses:
1. **MPS** (Apple Silicon GPU) - if available
2. **CUDA** (NVIDIA GPU) - if available
3. **CPU** - fallback (slower but works)

---

## 🔧 Troubleshooting

### Missing Model Files
**Error:** `FileNotFoundError: Translation model not found`

**Solution:** Ensure `translated_model/` folder exists with all required files.

### Missing QA Adapter
**Error:** `FileNotFoundError: QA model not found`

**Solution:** Ensure `qa_model/` folder exists with adapter files.

### PEFT Package Not Found
**Error:** `ModuleNotFoundError: No module named 'peft'`

**Solution:**
```bash
pip install peft>=0.7.0
```

### Out of Memory
**Solutions:**
- Close other applications
- Use CPU instead of GPU (automatic fallback)
- Reduce `max_new_tokens` in `main.py`

---

## 📚 Additional Documentation

- **PIPELINE_GUIDE.md** - Detailed pipeline documentation
- **verify_setup.py** - Check if setup is complete
- **translation_model.py** - Standalone translation testing
- **qa_model.py** - Standalone QA testing

---

## 🎯 Example Queries

Try these Darija questions:

```
شنو هي البرمجة؟
(What is programming?)

كيفاش نحسن الذاكرة ديالي؟
(How can I improve my memory?)

واش القهوة مضرة للصحة؟
(Is coffee bad for health?)

شنو هما فوائد الرياضة؟
(What are the benefits of exercise?)
```

---

## 🚀 Quick Start Summary

```bash
# 1. Setup environment
conda activate darija-translation
pip install -r requirements.txt

# 2. Verify setup (optional)
python verify_setup.py

# 3. Run pipeline
python main.py

# 4. Ask questions in Darija!
```

---

**Happy Querying! 🎉**
