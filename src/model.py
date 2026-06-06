"""
模型加载 — Qwen2.5 + QLoRA 4-bit / fp16 回退
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from .config import Config


def setup_model_and_tokenizer(config: Config):
    """初始化模型 + tokenizer + LoRA 适配器.

    Windows 兼容: 4-bit 失败时自动回退到 fp16.
    """

    # ── GPU 检查 ──
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available!\n"
            "  pip uninstall torch -y\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

    # ── Tokenizer ──
    print(f"Tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 模型 ──
    max_mem_for_model = int(gpu_mem_gb) - 6  # 预留 6GB 训练开销
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
                config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            model = prepare_model_for_kbit_training(model)
            print("  4-bit loaded OK!")
        except Exception as e:
            print(f"  4-bit failed: {e}")
            print("  Falling back to fp16...")
            use_4bit = False

    if not use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print("  fp16 loaded OK!")

    model.config.use_cache = False

    # ── LoRA ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
