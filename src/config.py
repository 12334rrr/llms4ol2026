"""
全局配置 + GPU 自动检测 + 模型推荐

支持的命令行:
    python train_ontology.py --config configs/a100_40gb.json
    python train_ontology.py --gpu auto          # 自动检测最佳配置
    python train_ontology.py --gpu a100          # 指定 GPU 类型
"""

import os
import json

# ═══════════════════════════════════════════════════════════
# 模型下载路径 & 国内镜像
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(MODELS_DIR, "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"  # 默认在线，离线时手动改
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def find_local_model(model_name: str) -> str:
    """
    查找模型。流程:
    1. 如果是本地存在的路径 → 直接返回
    2. 在 models/ 目录搜索 (多层递归)
    3. 都没找到 → 报错退出
    """
    # ── 1. 先处理各种路径形式 ──
    for path in [model_name, os.path.abspath(model_name)]:
        if os.path.isfile(os.path.join(path, "config.json")):
            print(f"[config] Using: {os.path.abspath(path)}")
            return os.path.abspath(path)

    # ── 2. 递归搜索 models/ ──
    if os.path.isdir(MODELS_DIR):
        for root, dirs, files in os.walk(MODELS_DIR):
            dirs[:] = [d for d in dirs if d not in ("blobs", "refs", ".locks")]
            if "config.json" in files:
                print(f"[config] Found model at: {root}")
                return root

    # ── 3. 没找到 ──
    model_short = model_name.split("/")[-1]
    print(f"[config] Model not found: {model_name}")
    print(f"[config] Place model at: models/{model_short}/config.json")
    print(f"[config] Or any subfolder of: models/")
    return model_name  # 返回原名, 让上层报错


# ═══════════════════════════════════════════════════════════
# GPU 检测 & 模型推荐
# ═══════════════════════════════════════════════════════════
GPU_INFO = None


def detect_gpu():
    """检测 GPU 并返回信息."""
    global GPU_INFO
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            GPU_INFO = {"name": name, "memory_gb": mem, "count": total}
            return GPU_INFO
    except:
        pass
    GPU_INFO = {"name": "unknown", "memory_gb": 0, "count": 0}
    return GPU_INFO


def gpu_to_preset(gpu_name: str, mem_gb: float) -> str:
    """根据 GPU 名称/显存返回推荐预设配置名."""
    name_lower = gpu_name.lower()
    # A100
    if "a100" in name_lower and mem_gb >= 38:
        return "a100_40gb"
    # A6000 / 4090
    if ("a6000" in name_lower or "4090" in name_lower) and mem_gb >= 40:
        return "a100_40gb"
    # V100
    if "v100" in name_lower and mem_gb >= 30:
        return "v100_32gb"
    # T4
    if "t4" in name_lower:
        return "t4_16gb"
    # 3090
    if "3090" in name_lower:
        return "rtx3090_24gb"
    # RTX 4090
    if "4090" in name_lower:
        return "a100_40gb"
    # 通用：按显存大小
    if mem_gb >= 38:
        return "a100_40gb"
    if mem_gb >= 28:
        return "v100_32gb"
    if mem_gb >= 20:
        return "rtx3090_24gb"
    return "t4_16gb"


def load_config(config_path=None, gpu_hint=None):
    """
    加载配置，优先级: config_path > gpu_hint > 自动检测.

    Returns: dict with all config values
    """
    gpu = detect_gpu()

    # 确定预设名
    if config_path and os.path.exists(config_path):
        preset_path = config_path
        preset_name = os.path.basename(config_path).replace(".json", "")
    elif gpu_hint:
        # 手动指定 GPU 类型
        preset_name = gpu_hint if gpu_hint != "auto" else gpu_to_preset(gpu["name"], gpu["memory_gb"])
        preset_path = os.path.join(SCRIPT_DIR, "configs", f"{preset_name}.json")
    elif gpu["memory_gb"] > 0:
        preset_name = gpu_to_preset(gpu["name"], gpu["memory_gb"])
        preset_path = os.path.join(SCRIPT_DIR, "configs", f"{preset_name}.json")
    else:
        preset_name = "rtx3090_24gb"  # fallback
        preset_path = os.path.join(SCRIPT_DIR, "configs", "rtx3090_24gb.json")

    # 加载预设
    preset = {}
    if os.path.exists(preset_path):
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset = json.load(f)

    # 构建完整配置
    config = Config()
    config.GPU_NAME = gpu["name"]
    config.GPU_MEMORY_GB = gpu["memory_gb"]
    config.GPU_COUNT = gpu["count"]
    config.PRESET = preset_name

    if preset:
        config.MODEL_NAME = preset.get("model_name", config.MODEL_NAME)
        config.USE_4BIT = preset.get("use_4bit", config.USE_4BIT)
        config.LORA_R = preset.get("lora_r", config.LORA_R)
        config.LORA_ALPHA = preset.get("lora_alpha", config.LORA_ALPHA)
        config.BATCH_SIZE = preset.get("batch_size", config.BATCH_SIZE)
        config.GRADIENT_ACCUMULATION = preset.get("gradient_accumulation", config.GRADIENT_ACCUMULATION)
        config.MAX_LENGTH = preset.get("max_length", config.MAX_LENGTH)
        config.LEARNING_RATE = preset.get("learning_rate", config.LEARNING_RATE)
        config.NUM_EPOCHS = preset.get("num_epochs", config.NUM_EPOCHS)

    # 命令行覆盖仍然生效 (由 train_ontology.py 处理)
    return config


# ═══════════════════════════════════════════════════════════
# 模型推荐表
# ═══════════════════════════════════════════════════════════
MODEL_RECOMMENDATIONS = {
    "a100_40gb": {
        "top": "Qwen/Qwen2.5-14B-Instruct",
        "good": "Qwen/Qwen2.5-7B-Instruct",
        "fast": "Qwen/Qwen2.5-3B-Instruct",
        "note": "14B with 4-bit ≈ 9GB VRAM, leaves 30GB for training"
    },
    "v100_32gb": {
        "top": "Qwen/Qwen2.5-7B-Instruct",
        "good": "Qwen/Qwen2.5-3B-Instruct",
        "note": "V100 不支持 bfloat16，设置 fp16=True"
    },
    "rtx3090_24gb": {
        "top": "Qwen/Qwen2.5-7B-Instruct",
        "good": "Qwen/Qwen2.5-3B-Instruct",
    },
    "t4_16gb": {
        "top": "Qwen/Qwen2.5-3B-Instruct",
        "good": "Qwen/Qwen2.5-7B-Instruct",
        "note": "小 batch_size=1, grad_accum=16"
    },
}


# ═══════════════════════════════════════════════════════════
# Config 类
# ═══════════════════════════════════════════════════════════
class Config:
    # ── 默认值 (会被 preset 覆盖) ──
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    MAX_LENGTH = 4096

    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    TRAIN_PATH = "data/train_task_a.json"
    VAL_SPLIT = 0.05
    OVERSAMPLE_THRESHOLD = 200
    OVERSAMPLE_MULTIPLIER = 5

    OUTPUT_DIR = "./output/ontology_lora"
    LOGGING_STEPS = 10
    SAVE_STEPS = 300
    EVAL_STEPS = 300
    SAVE_TOTAL_LIMIT = 3

    USE_4BIT = True
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    SEED = 42

    # GPU 信息 (运行时填充)
    GPU_NAME = "unknown"
    GPU_MEMORY_GB = 0
    GPU_COUNT = 0
    PRESET = "unknown"


# ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert in ontology learning and knowledge extraction. Your task is to analyze the given text and extract all ontology triples.

An ontology triple is a [subject, predicate, object] relationship. Extract these types:

1. **instance-of**: term/instance → type/class
2. **is-a**: subclass → superclass (taxonomic hierarchy)
3. **Non-taxonomic relations**: e.g. "part_of", "has part", "equivalent class", "disjoint with", "is defined by", "type", "exact match", "tree view", "see also", "located in", "derives from", "develops_from", "has role", "regulates", "broader", "domain", "range"

Important rules:
- Extract entities EXACTLY as they appear in the text
- Build a CONNECTED, COHERENT ontology graph
- Every instance/term must have at least one type (instance-of)
- Types must form a proper taxonomy (is-a), NO cycles
- Only extract relations EXPLICITLY stated or STRONGLY implied

Output ONLY valid JSON: {"triples": [["subject", "predicate", "object"], ...]}"""
