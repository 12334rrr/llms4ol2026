"""
模型加载 — Qwen2.5 + QLoRA 4-bit / fp16 回退
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from .config import Config, find_local_model


def _check_network() -> bool:
    """快速检测是否能连到 HuggingFace (优先国内镜像)."""
    import socket
    # 先试镜像，再试官方
    for host in ["hf-mirror.com", "huggingface.co"]:
        try:
            socket.create_connection((host, 443), timeout=3)
            return True
        except:
            continue
    return False


def _model_not_found_error(model_name: str, searched: list):
    """本地模型未找到时的处理."""
    model_short = model_name.split("/")[-1]
    has_net = _check_network()

    print("")
    print("=" * 60)
    if has_net:
        print("  📥 Model not local — downloading from mirror...")
    else:
        print("  ❌ MODEL NOT FOUND & SERVER OFFLINE")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Local path: models/{model_short}/")
    print("")

    if has_net:
        print("  Server has internet (mirror reachable), auto-downloading...")
        print(f"  Mirror: {os.environ.get('HF_ENDPOINT', 'huggingface.co')}")
        print("=" * 60)
        return  # 不抛异常，继续在线下载
    else:
        print("  ── On a machine WITH internet ──")
        print(f"    pip install huggingface_hub")
        print(f"    huggingface-cli download {model_name} --local-dir ./models/{model_short}")
        print("")
        print(f"  ── Then scp to this server ──")
        print(f"    scp -r models/{model_short} user@server:~/llms4ol2026/models/")
        print("=" * 60)
        raise RuntimeError(
            f"Model '{model_name}' not found locally and server is OFFLINE.\n"
            f"Download on a machine with internet and copy to models/{model_short}/"
        )


def setup_model_and_tokenizer(config: Config):
    """初始化模型 + tokenizer + LoRA 适配器."""

    # ── GPU 检查 ──
    if not torch.cuda.is_available():
        import subprocess
        driver_ver = 0
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
            driver_ver = int(out.stdout.strip().split(".")[0])
        except:
            pass
        cu_tag = "cu124" if driver_ver >= 545 else ("cu121" if driver_ver >= 525 else "cu118")
        raise RuntimeError(
            f"CUDA not available! (Driver: {driver_ver})\n"
            f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cu_tag}"
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

    # ── 查找本地模型 ──
    model_path = find_local_model(config.MODEL_NAME)
    is_local = os.path.isdir(model_path)

    if not is_local:
        _model_not_found_error(config.MODEL_NAME, [])
        # 如果没抛异常（=有网），继续尝试在线下载
        print("[model] Attempting download from HuggingFace (this may take a while)...")

    # ── Tokenizer ──
    print(f"Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        local_files_only=is_local
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 模型 ──
    max_mem_for_model = int(gpu_mem_gb) - 6
    max_memory = {0: f"{max_mem_for_model}GB", "cpu": "32GB"}
    model = None
    use_4bit = config.USE_4BIT

    if use_4bit:
        print(f"Trying 4-bit quantization (max={max_mem_for_model}GB)...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto", max_memory=max_memory,
                trust_remote_code=True, torch_dtype=torch.float16,
                local_files_only=is_local,
            )
            model = prepare_model_for_kbit_training(model)
            print("  4-bit loaded OK!")
        except Exception as e:
            print(f"  4-bit failed: {e}")
            print("  Falling back to fp16...")
            use_4bit = False

    if not use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto", max_memory=max_memory,
            trust_remote_code=True, torch_dtype=torch.float16,
            local_files_only=is_local,
        )
        print("  fp16 loaded OK!")

    model.config.use_cache = False

    # ── LoRA ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
