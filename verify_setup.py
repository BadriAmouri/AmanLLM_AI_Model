"""
Pipeline Setup Verification Script
Checks if all required components are ready before running main.py
"""

import os
import sys

print("="*70)
print("🔍 PIPELINE SETUP VERIFICATION")
print("="*70)

errors = []
warnings = []

# Check 1: Translation Model
print("\n[1/4] Checking Translation Model...")
if os.path.exists("./translated_model"):
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing = [f for f in required_files if not os.path.exists(f"./translated_model/{f}")]
    if missing:
        errors.append(f"Translation model missing files: {missing}")
        print(f"  ❌ Missing files: {missing}")
    else:
        print("  ✅ Translation model found and complete")
else:
    errors.append("Translation model directory './translated_model' not found")
    print("  ❌ Directory './translated_model' not found")

# Check 2: QA Model Adapter
print("\n[2/4] Checking QA Model Adapter...")
if os.path.exists("./qa_model"):
    required_files = ["adapter_config.json", "adapter_model.safetensors", "config.json"]
    missing = [f for f in required_files if not os.path.exists(f"./qa_model/{f}")]
    if missing:
        errors.append(f"QA model missing files: {missing}")
        print(f"  ❌ Missing files: {missing}")
    else:
        # Check adapter size
        adapter_path = "./qa_model/adapter_model.safetensors"
        size_mb = os.path.getsize(adapter_path) / (1024 * 1024)
        print(f"  ✅ QA model adapter found ({size_mb:.1f} MB)")
else:
    errors.append("QA model directory './qa_model' not found")
    print("  ❌ Directory './qa_model' not found")

# Check 3: Python Packages
print("\n[3/4] Checking Python Packages...")
required_packages = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "peft": "PEFT (for LoRA)",
}

for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {name} installed")
    except ImportError:
        errors.append(f"Package '{package}' not installed")
        print(f"  ❌ {name} NOT installed")

# Check 4: Device Availability
print("\n[4/4] Checking Compute Device...")
try:
    import torch
    if torch.backends.mps.is_available():
        device = "MPS (Apple Silicon GPU)"
        print(f"  ✅ {device} available")
    elif torch.cuda.is_available():
        device = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        print(f"  ✅ {device} available")
    else:
        device = "CPU"
        warnings.append("No GPU detected - will run on CPU (slower)")
        print(f"  ⚠️  {device} only (no GPU detected)")
except Exception as e:
    warnings.append(f"Could not check device: {e}")
    print(f"  ⚠️  Could not check device availability")

# Additional Info
print("\n" + "="*70)
print("ℹ️  ADDITIONAL INFORMATION")
print("="*70)

# Check cache directory
import pathlib
cache_dir = pathlib.Path.home() / ".cache" / "huggingface"
if cache_dir.exists():
    print(f"\n📁 Hugging Face cache: {cache_dir}")
    print("   (Base QA model will be cached here on first run)")
else:
    print(f"\n📁 Hugging Face cache will be created at: {cache_dir}")

print("\n⚠️  FIRST RUN WARNING:")
print("   The first run will download Qwen/Qwen2.5-3B-Instruct (~6GB)")
print("   This is a ONE-TIME download. Be patient!")

# Summary
print("\n" + "="*70)
print("📊 VERIFICATION SUMMARY")
print("="*70)

if errors:
    print(f"\n❌ Found {len(errors)} error(s):")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    print("\n🔧 Fix these errors before running main.py")
    sys.exit(1)
elif warnings:
    print(f"\n⚠️  Found {len(warnings)} warning(s):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")
    print("\n✅ Setup is OK but consider addressing warnings")
    print("✅ You can run: python main.py")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("✅ Ready to run: python main.py")

print("\n" + "="*70)
