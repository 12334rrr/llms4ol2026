"""
LLMs4OL 2026 — 一键式模型训练脚本
======================================
直接使用 Qwen2.5-7B-Instruct + QLoRA 4-bit 微调运行本体学习模型训练。
专为 RTX 3090 24GB 优化，兼容 Windows/Linux。

用法:
    python run_train.py                          # 使用所有默认值进行训练
    python run_train.py --epochs 5 --lr 1e-4     # 自定义超参数
    python run_train.py --model Qwen/Qwen2.5-3B-Instruct --no_4bit  # 使用较小模型，禁用 4bit
    python run_train.py --skip_check              # 跳过环境检查
    python run_train.py --dry_run                 # 完整试运行（无训练）

一键命令:
    python run_train.py
"""

import os
import sys
import json
import argparse
import time
import warnings
import platform
from datetime import datetime
from typing import Tuple, Optional

# ── 确保项目根目录位于 Python path 中 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Windows GBK 终端编码修复 ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # Python < 3.7 不支持 reconfigure

# 抑制良性警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ══════════════════════════════════════════════════════════════════════
# 训练配置
# ══════════════════════════════════════════════════════════════════════

class TrainConfig:
    """训练超参数 — 针对 6–24GB VRAM 优化，默认适配 6GB 环境。"""

    # ── 模型 ──
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    MAX_LENGTH = 2048                     # 6GB: 2048; >=16GB 可调至 4096

    # ── LoRA (QLoRA) ──
    LORA_R = 16                           # 6GB: 16; >=16GB 可调至 64
    LORA_ALPHA = 32                       # 通常为 r 的 2 倍
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # ── 训练超参数 ──
    NUM_EPOCHS = 3
    BATCH_SIZE = 1                        # 6GB: 1; >=16GB 可调至 4
    GRADIENT_ACCUMULATION = 16            # 有效 batch = 1 × 16 = 16
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # ── 数据 ──
    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "train_task_a.json")
    VAL_SPLIT = 0.05
    OVERSAMPLE_THRESHOLD = 200          # 出现次数 < 此值的谓词将进行过采样
    OVERSAMPLE_MULTIPLIER = 5

    # ── 输出 ──
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "ontology_qwen7b_lora")
    LOGGING_STEPS = 10
    SAVE_STEPS = 300
    EVAL_STEPS = 300
    SAVE_TOTAL_LIMIT = 3

    # ── 硬件 ──
    USE_4BIT = True
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    SEED = 42


# ══════════════════════════════════════════════════════════════════════
# 环境验证
# ══════════════════════════════════════════════════════════════════════

def print_header(title: str):
    """打印格式化标题。"""
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def print_step(step: int, title: str):
    """打印格式化步骤标题。"""
    print(f"\n  [{step}] {title}")
    print(f"  {'-' * 56}")


def check_environment(config: TrainConfig) -> bool:
    """
    运行前全面检查环境。返回 True 表示通过，False 表示未通过。
    问题汇总在一起，以便一次性修复。
    """
    print_header("环境检查")
    issues = []
    ok_list = []

    # ── Python 版本 ──
    py_ver = sys.version_info
    if py_ver >= (3, 9):
        ok_list.append(f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    else:
        issues.append(f"Python >= 3.9 是必需的，当前版本为 {py_ver.major}.{py_ver.minor}")

    # ── 操作系统 ──
    ok_list.append(f"操作系统: {platform.system()} {platform.release()}")

    # ── PyTorch ──
    try:
        import torch
        ok_list.append(f"PyTorch {torch.__version__}")
    except ImportError:
        issues.append("PyTorch 未安装。请运行: pip install torch>=2.0.0")
        torch = None

    # ── CUDA ──
    if torch is not None:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            ok_list.append(f"CUDA: {torch.version.cuda}")
            ok_list.append(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            if gpu_mem < 5.0:
                issues.append(f"GPU 显存 ({gpu_mem:.1f} GB) 严重不足，可能无法运行。4-bit QLoRA 建议至少 5GB VRAM")
            elif gpu_mem < 6.5:
                print(f"  [WARN] GPU 显存较小 ({gpu_mem:.1f} GB)，已自动启用低显存优化参数，继续运行...")
            elif gpu_mem < 16:
                print(f"  提示: GPU 显存较小 ({gpu_mem:.1f} GB)，已自动启用低显存优化参数")
        else:
            issues.append("未检测到 CUDA。请安装 CUDA 版 PyTorch（参见 https://pytorch.org）")

    # ── Transformers ──
    try:
        import transformers
        ok_list.append(f"Transformers {transformers.__version__}")
    except ImportError:
        issues.append("Transformers 未安装。请运行: pip install transformers>=4.40.0")

    # ── PEFT ──
    try:
        import peft
        ok_list.append(f"PEFT {peft.__version__}")
    except Exception as e:
        issues.append(f"PEFT 导入失败: {e}")

    # ── bitsandbytes ──
    try:
        import bitsandbytes
        ok_list.append(f"bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        if config.USE_4BIT:
            issues.append("bitsandbytes 未安装（4-bit QLoRA 必需）。请运行: pip install bitsandbytes>=0.43.0")
        else:
            ok_list.append("bitsandbytes: 跳过（4-bit 已禁用）")

    # ── Accelerate ──
    try:
        import accelerate
        ok_list.append(f"Accelerate {accelerate.__version__}")
    except ImportError:
        issues.append("Accelerate 未安装。请运行: pip install accelerate>=0.28.0")

    # ── 数据文件 ──
    if os.path.exists(config.TRAIN_PATH):
        with open(config.TRAIN_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ok_list.append(f"训练数据: {config.TRAIN_PATH} ({len(data)} 个样本)")
    else:
        issues.append(f"训练数据缺失: {config.TRAIN_PATH}")

    # ── 模型缓存 ──
    local_model = os.path.join(config.LOCAL_MODEL_DIR, "hub",
                               "models--Qwen--Qwen2.5-7B-Instruct")
    if os.path.isdir(local_model):
        total_size = 0
        for root, dirs, files in os.walk(local_model):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        ok_list.append(f"本地模型缓存: {local_model} ({total_size / 1024**3:.1f} GB)")
    else:
        ok_list.append("本地模型缓存: 未找到（将从 HuggingFace 下载）")

    # ── 打印结果 ──
    for item in ok_list:
        print(f"  [OK] {item}")
    if issues:
        print(f"\n  [FAIL] 发现问题 ({len(issues)}):")
        for issue in issues:
            print(f"    • {issue}")
        return False
    else:
        print(f"\n  [OK] 全部 {len(ok_list)} 项检查通过")
        return True


# ══════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════

def run_training(config: TrainConfig, dry_run: bool = False):
    """运行完整训练流水线。"""

    import numpy as np
    from collections import Counter

    # ── System Prompt ──
    SYSTEM_PROMPT = (
        "You are an expert in ontology learning and knowledge extraction. "
        "Your task is to analyze the given text and extract all ontology triples.\n\n"
        "An ontology triple is a [subject, predicate, object] relationship. Extract these types:\n\n"
        "1. **instance-of**: term/instance → type/class\n"
        '   ["temperature sensor", "instance-of", "sensor"]\n\n'
        "2. **is-a**: subclass → superclass (taxonomic hierarchy)\n"
        '   ["sensor", "is-a", "device"]\n\n'
        "3. **Non-taxonomic relations**: meaningful relationships between entities. "
        "Common predicates include:\n"
        '   - part-whole: "part_of", "has part"\n'
        '   - equivalence: "equivalent class", "exact match", "same as"\n'
        '   - disjointness: "disjoint with"\n'
        '   - definitional: "is defined by", "type"\n'
        '   - process/role: "has role", "regulates", "derives from", "develops_from"\n'
        '   - location: "located in"\n'
        '   - other: "tree view", "database_cross_reference", "see also", "broader"\n\n'
        "Important rules:\n"
        "- Extract entities EXACTLY as they appear in the text\n"
        "- Build a CONNECTED, COHERENT ontology graph\n"
        "- Ensure every instance/term has at least one type via instance-of\n"
        "- Ensure types form a proper taxonomy hierarchy via is-a\n"
        "- The taxonomy must NOT have cycles\n"
        "- Only extract relations EXPLICITLY stated or STRONGLY implied\n\n"
        'Output ONLY valid JSON: {"triples": [["subject", "predicate", "object"], ...]}'
    )

    # ═══════════════════════════ 步骤 1: 数据 ═══════════════════════════
    print_header("步骤 1/4: 加载与准备数据")

    print(f"  来源: {config.TRAIN_PATH}")
    with open(config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"  已加载样本数: {len(raw_data)}")

    # 统计谓词频率
    pred_counter = Counter()
    for sample in raw_data:
        for t in sample.get("primitive-ontology-triples", []):
            if len(t) == 3:
                pred_counter[t[1]] += 1

    print(f"  唯一谓词类型: {len(pred_counter)}")
    print(f"  三元组总数: {sum(pred_counter.values())}")
    print(f"  前 5 种谓词:")
    for pred, cnt in pred_counter.most_common(5):
        pct = 100.0 * cnt / sum(pred_counter.values())
        print(f"    {pred}: {cnt} ({pct:.1f}%)")

    # ChatML 格式化
    formatted = []
    skipped = 0
    for sample in raw_data:
        triples = sample.get("primitive-ontology-triples", [])
        if not triples:
            skipped += 1
            continue

        target = json.dumps({"triples": triples}, ensure_ascii=False)
        full_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{sample['context']}<|im_end|>\n"
            f"<|im_start|>assistant\n{target}<|im_end|>"
        )

        has_rare = any(
            pred_counter[t[1]] < config.OVERSAMPLE_THRESHOLD
            for t in triples if len(t) == 3
        )

        formatted.append({
            "text": full_text,
            "num_triples": len(triples),
            "has_rare_predicate": has_rare,
        })
    if skipped:
        print(f"  已跳过空三元组样本数: {skipped}")

    # 对含低频谓词的样本进行过采样
    rare_samples = [d for d in formatted if d["has_rare_predicate"]]
    n_before = len(formatted)
    if rare_samples:
        print(f"  低频谓词样本数: {len(rare_samples)}（阈值 < {config.OVERSAMPLE_THRESHOLD}）")
        for _ in range(config.OVERSAMPLE_MULTIPLIER - 1):
            formatted.extend(rare_samples)
        print(f"  过采样后 ({config.OVERSAMPLE_MULTIPLIER}x): {len(formatted)} (曾为 {n_before})")
    else:
        print("  无需过采样（所有谓词均超过阈值）")

    # 训练/验证集划分
    np.random.seed(config.SEED)
    indices = np.random.permutation(len(formatted))
    val_size = max(1, int(len(formatted) * config.VAL_SPLIT))
    train_data = [formatted[i] for i in indices[val_size:].tolist()]
    val_data = [formatted[i] for i in indices[:val_size].tolist()]
    print(f"  训练集: {len(train_data)}  验证集: {len(val_data)} ({(100*len(val_data)/len(formatted)):.1f}%)")

    if dry_run:
        print("\n  >>> 试运行模式: 跳过模型加载与训练 <<<")
        return

    # ── 重型导入（仅在真正训练时需要）──
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        TrainingArguments,
        Trainer,
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )

    # ═══════════════════════════ 步骤 2: 模型 ═══════════════════════════
    print_header("步骤 2/4: 加载模型与 Tokenizer")

    # 在本地缓存中定位模型（动态匹配模型名）
    model_path = config.MODEL_NAME
    # 将 HuggingFace 模型 ID 转换为本地缓存路径格式: "Qwen/Qwen2.5-7B-Instruct" → "models--Qwen--Qwen2.5-7B-Instruct"
    cache_dir_name = "models--" + config.MODEL_NAME.replace("/", "--")
    local_snapshot = os.path.join(config.LOCAL_MODEL_DIR, "hub", cache_dir_name, "snapshots")
    if os.path.isdir(local_snapshot):
        snapshots = sorted(os.listdir(local_snapshot))
        if snapshots:
            model_path = os.path.join(local_snapshot, snapshots[-1])
            print(f"  本地模型: {model_path}")

    # GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

    # Tokenizer
    print(f"  加载 tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if os.path.isdir(model_path) else config.MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Tokenizer 词表大小: {tokenizer.vocab_size}")

    # 模型（4-bit 优先 → fp16 回退）
    # 4-bit 模型权重约 3.5-4GB；6GB 卡预留 1GB 给训练开销，其余全给模型
    max_mem_for_model = max(3.0, gpu_mem_gb - 1.0)
    max_memory = {0: f"{max_mem_for_model}GB", "cpu": "32GB"}
    use_4bit = config.USE_4BIT
    model = None

    if use_4bit:
        print(f"  尝试 4-bit 量化 (max GPU = {max_mem_for_model} GB)...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_path if os.path.isdir(model_path) else config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            model = prepare_model_for_kbit_training(model)
            elapsed = time.time() - t0
            print(f"  [OK] 4-bit 模型加载成功! (耗时 {elapsed:.1f}s)")
        except Exception as e:
            print(f"  [FAIL] 4-bit 加载失败: {e}")
            print("  -> 回退至 fp16...")
            use_4bit = False

    if not use_4bit:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path if os.path.isdir(model_path) else config.MODEL_NAME,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        elapsed = time.time() - t0
        print(f"  [OK] fp16 模型加载成功! (耗时 {elapsed:.1f}s)")

    model.config.use_cache = False

    # LoRA
    print(f"  应用 LoRA (r={config.LORA_R}, alpha={config.LORA_ALPHA})...")
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

    # VRAM 报告
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  GPU 显存: 已分配 {allocated:.1f} GB, 已预留 {reserved:.1f} GB")

    # ═══════════════════════════ 步骤 3: 数据集 ═══════════════════════════
    print_header("步骤 3/4: 创建数据集与 DataCollator")

    class OntologyDataset(Dataset):
        def __init__(self, data, tokenizer, max_length: int = 4096):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            tok = self.tokenizer(
                self.data[idx]["text"],
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            tok["labels"] = tok["input_ids"].copy()
            return tok

    class DataCollator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = pad_token_id

        def __call__(self, batch):
            max_len = max(len(b["input_ids"]) for b in batch)
            input_ids, attn, labels = [], [], []
            for b in batch:
                p = max_len - len(b["input_ids"])
                input_ids.append(b["input_ids"] + [self.pad_token_id] * p)
                attn.append([1] * len(b["input_ids"]) + [0] * p)
                labels.append(b["labels"] + [-100] * p)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    train_ds = OntologyDataset(train_data, tokenizer, config.MAX_LENGTH)
    val_ds = OntologyDataset(val_data, tokenizer, config.MAX_LENGTH)
    collator = DataCollator(tokenizer.pad_token_id)
    print(f"  训练数据集大小: {len(train_ds)}  验证数据集大小: {len(val_ds)}")

    # ═══════════════════════════ 步骤 4: 训练 ═══════════════════════════
    effective_bs = config.BATCH_SIZE * config.GRADIENT_ACCUMULATION
    print_header("步骤 4/4: 训练")
    print(f"  Epochs:              {config.NUM_EPOCHS}")
    print(f"  Batch 大小:          {config.BATCH_SIZE} × {config.GRADIENT_ACCUMULATION} = {effective_bs}")
    print(f"  学习率:               {config.LEARNING_RATE}")
    print(f"  预热比例:            {config.WARMUP_RATIO}")
    print(f"  权重衰减:            {config.WEIGHT_DECAY}")
    print(f"  最大长度:            {config.MAX_LENGTH}")
    print(f"  输出目录:            {config.OUTPUT_DIR}")
    print(f"  4-bit QLoRA:         {'是' if use_4bit else '否 (fp16)'}")
    print()

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        warmup_ratio=config.WARMUP_RATIO,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.FP16,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_dir=os.path.join(config.OUTPUT_DIR, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    t_start = time.time()
    print("  开始训练...\n")
    train_result = trainer.train()
    train_time = time.time() - t_start

    # ── 训练指标 ──
    print(f"\n  训练耗时: {train_time/60:.1f} 分钟 ({train_time:.0f}s)")
    print(f"  训练损失: {train_result.training_loss:.4f}" if train_result.training_loss else "")
    metrics = trainer.evaluate()
    print(f"  验证损失: {metrics.get('eval_loss', 'N/A'):.4f}" if 'eval_loss' in metrics else "")

    # ── 保存最终模型 ──
    print_header("保存模型")
    final_path = os.path.join(config.OUTPUT_DIR, "final_model")
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # 保存训练配置 snapshot
    config_snapshot = {
        "model_name": config.MODEL_NAME,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "num_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation": config.GRADIENT_ACCUMULATION,
        "effective_batch_size": effective_bs,
        "learning_rate": config.LEARNING_RATE,
        "max_length": config.MAX_LENGTH,
        "use_4bit": use_4bit,
        "train_time_seconds": train_time,
        "train_loss": train_result.training_loss,
        "eval_loss": metrics.get("eval_loss"),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(final_path, "training_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    print(f"  [OK] 模型已保存至: {final_path}")
    print(f"  [OK] 训练配置已保存至: {os.path.join(final_path, 'training_config.json')}")

    # ── 摘要 ──
    print_header("训练完成！")
    print(f"  最终模型:          {final_path}")
    print(f"  总训练时间:        {train_time/60:.1f} 分钟")
    print(f"  最佳验证损失:      {metrics.get('eval_loss', 'N/A'):.4f}" if 'eval_loss' in metrics else "")
    print(f"\n  下一步 — 运行预测:")
    print(f"    python predict_ontology.py --model_path {final_path}")
    print(f"  或者 — 运行评估:")
    print(f"    python main.py evaluate")


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def load_config_file(config_path: str) -> dict:
    """加载 config.json 配置文件，不存在则返回空 dict。"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARN] 配置文件读取失败 ({config_path}): {e}")
    return {}


def detect_local_models(models_dir: str) -> dict:
    """扫描本地已缓存的模型，返回 {model_id: snapshot_path}。"""
    local = {}
    hub_dir = os.path.join(models_dir, "hub")
    if not os.path.isdir(hub_dir):
        return local
    for entry in os.listdir(hub_dir):
        if entry.startswith("models--"):
            # models--Qwen--Qwen2.5-7B-Instruct → Qwen/Qwen2.5-7B-Instruct
            model_id = entry.replace("models--", "").replace("--", "/")
            snapshots_dir = os.path.join(hub_dir, entry, "snapshots")
            if os.path.isdir(snapshots_dir):
                snaps = sorted(os.listdir(snapshots_dir))
                if snaps:
                    local[model_id] = os.path.join(snapshots_dir, snaps[-1])
    return local


def auto_select_model(gpu_mem_gb: float, local_models: dict) -> str:
    """根据 GPU 显存自动选择最合适的模型（优先本地已有模型）。"""
    # 按显存推荐: 7B 需 ≥7GB, 3B 需 ≥3.5GB, 1.5B 需 ≥2.5GB
    candidates = []
    for model_id in local_models:
        if "7B" in model_id and gpu_mem_gb >= 7.0:
            candidates.append((7, model_id))
        elif "3B" in model_id and gpu_mem_gb >= 3.5:
            candidates.append((3, model_id))
        elif "1.5B" in model_id and gpu_mem_gb >= 2.5:
            candidates.append((1.5, model_id))
    # 选本地最大的能跑得动的
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # 本地没有，按显存推荐下载
    if gpu_mem_gb >= 7.0:
        return "Qwen/Qwen2.5-7B-Instruct"
    elif gpu_mem_gb >= 3.5:
        return "Qwen/Qwen2.5-3B-Instruct"
    else:
        return "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    # ── 读取配置文件 ──
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    cfg_file = load_config_file(config_path)

    # 配置文件中的 available_models
    known_models = cfg_file.get("available_models", {
        "Qwen/Qwen2.5-7B-Instruct":  "7B 参数",
        "Qwen/Qwen2.5-3B-Instruct":  "3B 参数",
        "Qwen/Qwen2.5-1.5B-Instruct": "1.5B 参数",
    })

    # 从配置文件取默认值（命令行可覆盖）
    cfg_model = cfg_file.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
    cfg_output = cfg_file.get("output_dir", "./output/ontology_qwen_lora")

    # ── 检测本地模型 & GPU → 自动选模型 ──
    local_models = detect_local_models(os.path.join(PROJECT_ROOT, "models"))

    # 提前探测 GPU（仅用于自动选模型，失败忽略）
    auto_model = cfg_model
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            auto_model = auto_select_model(gpu_mem_gb, local_models)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="LLMs4OL 2026 — 使用 QLoRA 进行本体学习模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
配置文件: config.json（修改 model_name 即可切换模型）

本地已缓存模型:
""" + "\n".join(f"  • {k}  →  {v}" for k, v in local_models.items()) + f"""

可用模型列表:
""" + "\n".join(f"  • {k}  —  {v}" for k, v in known_models.items()) + f"""

示例:
  python run_train.py                              # 自动选模型 + 默认参数
  python run_train.py --model Qwen/Qwen2.5-3B-Instruct   # 指定模型
  python run_train.py --epochs 5 --lr 1e-4         # 自定义超参数
  python run_train.py --dry_run                    # 仅展示流程，不训练
  python run_train.py --skip_check                 # 跳过环境检查
        """,
    )

    # 核心参数
    core = parser.add_argument_group("核心参数")
    core.add_argument("--epochs", type=int, default=cfg_file.get("epochs", 3),
                      help="训练轮数（默认: 3）")
    core.add_argument("--batch_size", type=int, default=cfg_file.get("batch_size", 1),
                      help="每设备 batch 大小（默认: 1）")
    core.add_argument("--lr", type=float, default=cfg_file.get("learning_rate", 2e-4),
                      help="学习率（默认: 2e-4）")
    core.add_argument("--lora_r", type=int, default=cfg_file.get("lora_r", 16),
                      help="LoRA 秩（默认: 16）")

    # 模型参数
    model_grp = parser.add_argument_group("模型参数")
    model_grp.add_argument("--model", type=str, default=auto_model,
                           help=f"模型名称（自动检测: {auto_model}）")
    model_grp.add_argument("--output_dir", type=str, default=cfg_output,
                           help=f"输出目录（默认: {cfg_output}）")
    model_grp.add_argument("--no_4bit", action="store_true",
                           help="禁用 4-bit 量化")
    model_grp.add_argument("--max_length", type=int,
                           default=cfg_file.get("max_length", 2048),
                           help="最大序列长度（默认: 2048）")

    # 行为参数
    behavior = parser.add_argument_group("行为")
    behavior.add_argument("--skip_check", action="store_true",
                          help="跳过环境检查")
    behavior.add_argument("--dry_run", action="store_true",
                          help="仅展示训练流程，不实际训练")
    behavior.add_argument("--seed", type=int,
                           default=cfg_file.get("seed", 42),
                           help="随机种子（默认: 42）")

    args = parser.parse_args()

    # ── 构建配置 ──
    config = TrainConfig()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.LORA_R = args.lora_r
    config.MODEL_NAME = args.model
    config.OUTPUT_DIR = args.output_dir
    config.MAX_LENGTH = args.max_length
    config.SEED = args.seed
    config.LORA_ALPHA = cfg_file.get("lora_alpha", 32)
    config.GRADIENT_ACCUMULATION = cfg_file.get("gradient_accumulation", 16)
    if args.no_4bit or not cfg_file.get("use_4bit", True):
        config.USE_4BIT = False

    # ── 项目根目录中的 HF 缓存 ──
    os.environ.setdefault("HF_HOME", config.LOCAL_MODEL_DIR)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(config.LOCAL_MODEL_DIR, "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # ── 打印 banner ──
    model_short = config.MODEL_NAME.split("/")[-1] if "/" in config.MODEL_NAME else config.MODEL_NAME
    print(f"""
+=======================================================================+
|        LLMs4OL 2026 — End-to-End Ontology Learning Training           |
|        {model_short} + QLoRA 4-bit Fine-tuning                  |
+=======================================================================+
""")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  模型: {config.MODEL_NAME}")
    print(f"  输出: {config.OUTPUT_DIR}")
    print(f"  Epochs: {config.NUM_EPOCHS}  Batch: {config.BATCH_SIZE}  LR: {config.LEARNING_RATE}")
    print(f"  LoRA r: {config.LORA_R}  4-bit: {'是' if config.USE_4BIT else '否'}")

    # ── 显示本地模型 ──
    if local_models:
        print(f"\n  已缓存模型:")
        for mid, path in local_models.items():
            tag = " ★ 当前使用" if mid == config.MODEL_NAME else ""
            print(f"    • {mid}{tag}")
    else:
        print(f"\n  本地无缓存模型，将从 HuggingFace 下载 {config.MODEL_NAME}")

    # ── 环境检查 ──
    if not args.skip_check:
        if not check_environment(config):
            print("\n  提示: 使用 --skip_check 跳过环境检查（不推荐）")
            print("  或者修复上述问题后重试。")
            sys.exit(1)

    # ── 训练 ──
    try:
        run_training(config, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\n  训练已由用户中断。")
        sys.exit(130)
    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            print(f"\n  [FAIL] CUDA 显存不足!")
            print(f"    已尝试的 batch 大小: {config.BATCH_SIZE}")
            print(f"    当前 GPU 显存较小，建议依次尝试:")
            print(f"      1. 减小序列长度:           --max_length 1024")
            print(f"      2. 切换至 1.5B 模型:       --model Qwen/Qwen2.5-1.5B-Instruct")
            print(f"      3. 降低 LoRA rank:         --lora_r 8")
            print(f"      4. 增大梯度累积补偿:       --batch_size 1（并设 GRADIENT_ACCUMULATION=32）")
        else:
            print(f"\n  [FAIL] 训练错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [FAIL] 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
