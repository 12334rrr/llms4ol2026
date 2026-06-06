"""
GPU 诊断 + 模型推荐

Usage: python scripts/check_gpu.py
"""

import os, sys, subprocess

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(SCRIPT_DIR, "models")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def nvidia_smi():
    print("=" * 55)
    print("1. nvidia-smi")
    print("=" * 55)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                idx, name, mem_total, mem_free, util, temp, driver = line.split(", ")
                name = name.strip()
                mem_used = int(float(mem_total)) - int(float(mem_free))
                print(f"  GPU {idx}: {name}")
                print(f"    VRAM    : {mem_used}/{int(float(mem_total))} MB used ({mem_free} MB free)")
                print(f"    Util    : {util}%")
                print(f"    Temp    : {temp}C")
                print(f"    Driver  : {driver}")
        else:
            print("  Error:", r.stderr)
    except FileNotFoundError:
        print("  nvidia-smi not found!")
    except Exception as e:
        print(f"  Error: {e}")
    print()


def pytorch_gpu():
    print("=" * 55)
    print("2. PyTorch CUDA")
    print("=" * 55)
    try:
        import torch
        print(f"  PyTorch     : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA        : ✅ v{torch.version.cuda}")
            print(f"  cuDNN       : {torch.backends.cudnn.version()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                mem = p.total_mem / 1024**3
                print(f"  GPU {i}       : {p.name} ({mem:.1f} GB)")
                print(f"    Compute   : sm_{p.major}{p.minor}")
                print(f"    SMs       : {p.multi_processor_count}")
        else:
            print("  CUDA        : ❌ NOT AVAILABLE")
            print("  →  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  PyTorch not installed!")
    print()


def model_recommend():
    print("=" * 55)
    print("3. Model Recommendation")
    print("=" * 55)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  Cannot recommend — CUDA not available")
            return

        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        count = torch.cuda.device_count()
        name_l = name.lower()

        # 推荐逻辑
        if "a100" in name_l and mem >= 38:
            rec = {
                "model": "Qwen/Qwen2.5-14B-Instruct 🥇",
                "method": "4-bit QLoRA, batch=4×4",
                "alt": "Qwen/Qwen2.5-7B-Instruct (faster, also excellent)",
            }
        elif "a100" in name_l:
            rec = {
                "model": "Qwen/Qwen2.5-7B-Instruct 🥇",
                "method": "4-bit QLoRA, batch=4×4",
                "alt": "Qwen/Qwen2.5-3B-Instruct (faster iteration)",
            }
        elif "v100" in name_l:
            rec = {
                "model": "Qwen/Qwen2.5-7B-Instruct 🥇",
                "method": "4-bit QLoRA, batch=4×4",
                "note": "V100 不支持 bf16，已自动设置 fp16=True",
            }
        elif mem >= 38:
            rec = {
                "model": "Qwen/Qwen2.5-14B-Instruct 🥇",
                "method": "4-bit QLoRA, batch=4×4",
            }
        elif mem >= 28:
            rec = {
                "model": "Qwen/Qwen2.5-7B-Instruct 🥇",
                "method": "4-bit QLoRA, batch=4×4",
            }
        elif mem >= 20:
            rec = {
                "model": "Qwen/Qwen2.5-7B-Instruct 🥇",
                "method": "4-bit QLoRA, batch=4×4 (RTX 3090)",
            }
        else:
            rec = {
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "method": "4-bit QLoRA, batch=1×16 (T4/laptop)",
            }

        print(f"  GPU        : {name} ({mem:.1f} GB × {count})")
        print(f"  Best Model : {rec['model']}")
        print(f"  Method     : {rec['method']}")
        if "alt" in rec:
            print(f"  Also Good  : {rec['alt']}")
        if "note" in rec:
            print(f"  Note       : {rec['note']}")

        print(f"\n  🚀 Ready to train:")
        print(f"     python scripts/train.py --gpu auto")
        if count > 1:
            print(f"     python scripts/train.py --gpu auto --multi_gpu")
    except Exception as e:
        print(f"  Error: {e}")
    print()


if __name__ == "__main__":
    nvidia_smi()
    pytorch_gpu()
    model_recommend()
    print("=" * 55)
    print("Done!")
