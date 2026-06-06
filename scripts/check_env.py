"""
完整环境诊断 — 镜像 + GPU + 依赖 + 模型加载测试

Usage: python scripts/check_env.py
"""

import os, sys

# 先设置镜像
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(MODELS_DIR, "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from collections import defaultdict

PASS = "✓"
FAIL = "✗"


def check_cuda():
    print("=" * 50); print("1. CUDA / GPU"); print("=" * 50)
    print(f"  PyTorch: {torch.__version__}")

    if not torch.cuda.is_available():
        print(f"  CUDA available: {FAIL} ** CRITICAL **")
        print("  → PyTorch is CPU-only. Reinstall with CUDA:")
        print("    pip uninstall torch -y")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False

    print(f"  CUDA: {PASS} (v{torch.version.cuda})")
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {gpu} ({mem:.1f} GB)")
    print(f"  VRAM free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
    return True


def check_bitsandbytes():
    print("=" * 50); print("2. bitsandbytes (4-bit)"); print("=" * 50)
    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes: {bnb.__version__}")

        if not torch.cuda.is_available():
            print(f"  CUDA required but not available → {FAIL}")
            return False

        # Quick test: check if bnb can see CUDA
        try:
            import bitsandbytes.functional as BF
            print(f"  CUDA functional: {PASS}")
            return True
        except Exception as e:
            print(f"  functional test: {FAIL} — {e}")
            print("  → Use --no_4bit flag")
            return False
    except ImportError:
        print(f"  Not installed {FAIL}")
        return False


def check_libraries():
    print("=" * 50); print("3. Libraries"); print("=" * 50)
    libs = {
        "transformers": None, "peft": None, "accelerate": None,
        "datasets": None, "sentencepiece": None, "numpy": None, "tqdm": None,
    }
    for lib in libs:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "?")
            print(f"  {lib:15s}: {PASS} {ver}")
        except ImportError:
            print(f"  {lib:15s}: {FAIL}")


def check_data():
    print("=" * 50); print("4. Data"); print("=" * 50)
    import json
    for name in ["data/train_task_a.json", "data/test_task_a_input.json"]:
        path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"  {name}: {PASS} ({len(data)} samples)")
        else:
            print(f"  {name}: {FAIL}")


def test_model_load():
    print("=" * 50); print("5. Model Load Test (0.5B)"); print("=" * 50)

    if not torch.cuda.is_available():
        print("  Skipped — no CUDA")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    test_model = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"  Testing: {test_model}")

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            test_model, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
        )
        print(f"  4-bit load: {PASS}")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  4-bit load: {FAIL} — {str(e)[:100]}")
        print("  → Use --no_4bit --model Qwen/Qwen2.5-3B-Instruct")


if __name__ == "__main__":
    print(f"Models dir : {MODELS_DIR}")
    print(f"HF mirror : {os.environ.get('HF_ENDPOINT', 'default')}\n")

    cuda_ok = check_cuda()
    check_bitsandbytes()
    check_libraries()
    check_data()
    if cuda_ok:
        test_model_load()

    print("\n" + "=" * 50)
    if cuda_ok:
        print("SUMMARY: Environment OK — ready to train!")
        print("  python scripts/train.py --gpu auto --epochs 3")
    else:
        print("SUMMARY: ** CUDA NOT AVAILABLE **")
        print("  Fix PyTorch first, then rerun check_env.py")
