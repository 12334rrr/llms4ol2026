"""
训练报告生成模块 — 自动记录训练过程，生成 Markdown + JSON 报告

功能:
- TrainerCallback 自动记录 loss, lr, step, epoch
- 训练结束后生成 report.md + report.json
- 包含：配置摘要、loss 曲线(ascii)、耗时、显存使用
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

import torch
from transformers import TrainerCallback


# ═══════════════════════════════════════════════════════════
# Trainer Callback — 自动记录训练指标
# ═══════════════════════════════════════════════════════════
class ReportCallback(TrainerCallback):
    """HuggingFace Trainer 回调，记录每一步的 loss 和 GPU 信息."""

    def __init__(self, config, output_dir: str):
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.start_time = None
        self.step_start_time = None

        # 存储所有指标
        self.train_losses: List[dict] = []   # [{step, loss, lr, time_s}, ...]
        self.eval_losses: List[dict] = []    # [{step, eval_loss}, ...]
        self.gpu_samples: List[dict] = []    # 定期 GPU 快照

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次 logging_steps 触发，记录 loss/lr."""
        if logs is None:
            return

        elapsed = time.time() - self.start_time
        step_time = time.time() - self.step_start_time if self.step_start_time else 0

        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 2) if state.epoch else 0,
            "time_s": round(elapsed, 1),
            "step_time_s": round(step_time, 2),
        }

        # 训练 loss
        if "loss" in logs:
            entry["loss"] = round(logs["loss"], 6)
        if "learning_rate" in logs:
            entry["lr"] = logs["learning_rate"]

        if "loss" in logs or "learning_rate" in logs:
            self.train_losses.append(entry)

        # 验证 loss
        if "eval_loss" in logs:
            self.eval_losses.append({
                "step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0,
                "eval_loss": round(logs["eval_loss"], 6),
                "time_s": round(elapsed, 1),
            })

        # 定期 GPU 快照 (每 50 步)
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            self.gpu_samples.append({
                "step": state.global_step,
                "gpu_allocated_gb": round(mem_allocated, 2),
                "gpu_reserved_gb": round(mem_reserved, 2),
                "time_s": round(elapsed, 1),
            })

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        self.total_time_s = total_time
        self.final_step = state.global_step
        self.best_metric = state.best_metric if hasattr(state, 'best_metric') else None
        self.best_model_checkpoint = state.best_model_checkpoint if hasattr(state, 'best_model_checkpoint') else None


# ═══════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════
def generate_report(config, report_callback: ReportCallback, output_dir: str):
    """
    生成训练报告 (Markdown + JSON).

    Args:
        config: 训练配置
        report_callback: 训练结束后的回调实例
        output_dir: 报告保存目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 收集数据 ──
    total_time = getattr(report_callback, "total_time_s", 0)
    final_step = getattr(report_callback, "final_step", 0)
    best_metric = getattr(report_callback, "best_metric", None)
    best_ckpt = getattr(report_callback, "best_model_checkpoint", None)
    train_losses = report_callback.train_losses
    eval_losses = report_callback.eval_losses
    gpu_samples = report_callback.gpu_samples

    # ── 生成报告数据 ──
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 1),
            "total_time_human": str(timedelta(seconds=int(total_time))),
            "final_step": final_step,
            "best_eval_loss": best_metric,
        },
        "config": {
            "model": config.MODEL_NAME,
            "preset": getattr(config, "PRESET", "N/A"),
            "gpu": config.GPU_NAME,
            "gpu_memory_gb": config.GPU_MEMORY_GB,
            "gpu_count": config.GPU_COUNT,
            "use_4bit": config.USE_4BIT,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION,
            "effective_batch": config.BATCH_SIZE * config.GRADIENT_ACCUMULATION,
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS,
            "max_length": config.MAX_LENGTH,
        },
        "training": {
            "num_train_losses": len(train_losses),
            "num_eval_losses": len(eval_losses),
            "num_gpu_samples": len(gpu_samples),
            "final_train_loss": train_losses[-1]["loss"] if train_losses else None,
            "final_eval_loss": eval_losses[-1]["eval_loss"] if eval_losses else None,
        },
        "gpu": {
            "peak_allocated_gb": max((s["gpu_allocated_gb"] for s in gpu_samples), default=0),
            "avg_allocated_gb": round(sum(s["gpu_allocated_gb"] for s in gpu_samples) / len(gpu_samples), 2) if gpu_samples else 0,
        },
    }

    # ── 保存 JSON 报告 ──
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 生成 Markdown 报告 ──
    md = _build_markdown(config, report, train_losses, eval_losses, gpu_samples, total_time)
    md_path = os.path.join(output_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n📊 Report saved:")
    print(f"   {json_path}")
    print(f"   {md_path}")
    return report


def _ascii_loss_plot(losses: List[dict], key: str, width: int = 50, height: int = 10) -> str:
    """用 ASCII 字符画 loss 曲线."""
    if len(losses) < 2:
        return "  (not enough data)"

    values = [l[key] for l in losses if key in l]
    if len(values) < 2:
        return "  (not enough data)"

    min_v, max_v = min(values), max(values)
    rng = max_v - min_v if max_v > min_v else 1

    # 降采样到 width 个点
    step = max(1, len(values) // width)
    sampled = values[::step][:width]

    # 缩放到 height
    scaled = [int((v - min_v) / rng * (height - 1)) for v in sampled]

    # 画图 (Y轴反转，顶部=最小值)
    lines = []
    for y in range(height - 1, -1, -1):
        line = ""
        for x in range(len(scaled)):
            line += "█" if scaled[x] >= y else " "
        val_at_y = min_v + rng * (height - 1 - y) / (height - 1)
        label = f" {val_at_y:.4f}" if y in (0, height // 2, height - 1) else ""
        lines.append(f"  │{line}│{label}")
    lines.append(f"  └{'─' * len(scaled)}┘")
    lines.append(f"   {'Step →':<{len(scaled)}}")

    return "\n".join(lines)


def _build_markdown(config, report, train_losses, eval_losses, gpu_samples, total_time) -> str:
    """构建 Markdown 报告."""

    r = report

    lines = []
    lines.append("# 🏆 LLMs4OL 2026 — Training Report")
    lines.append("")
    lines.append(f"**Generated:** {r['metadata']['generated_at']}")
    lines.append(f"**Total Time:** {r['metadata']['total_time_human']}")
    lines.append("")

    # ── 配置 ──
    lines.append("## ⚙️ Configuration")
    lines.append("")
    cfg = r["config"]
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Model | `{cfg['model']}` |")
    lines.append(f"| GPU | {cfg['gpu']} ({cfg['gpu_memory_gb']:.0f} GB × {cfg['gpu_count']}) |")
    lines.append(f"| Preset | `{cfg['preset']}` |")
    lines.append(f"| Quantization | {'4-bit QLoRA' if cfg['use_4bit'] else 'fp16'} |")
    lines.append(f"| LoRA | r={cfg['lora_r']}, α={cfg['lora_alpha']} |")
    lines.append(f"| Batch Size | {cfg['batch_size']} × {cfg['gradient_accumulation']} = {cfg['effective_batch']} |")
    lines.append(f"| Learning Rate | {cfg['learning_rate']} |")
    lines.append(f"| Epochs | {cfg['num_epochs']} |")
    lines.append(f"| Max Length | {cfg['max_length']} tokens |")
    lines.append("")

    # ── 训练结果 ──
    lines.append("## 📈 Training Results")
    lines.append("")
    tr = r["training"]
    best_eval = r["metadata"]["best_eval_loss"]

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Final Step | {r['metadata']['final_step']} |")
    lines.append(f"| Final Train Loss | {tr['final_train_loss']:.6f}" if tr['final_train_loss'] else "| Final Train Loss | N/A |")
    lines.append(f"| Final Eval Loss | {tr['final_eval_loss']:.6f}" if tr['final_eval_loss'] else "| Final Eval Loss | N/A |")
    lines.append(f"| Best Eval Loss | {best_eval:.6f}" if best_eval else "| Best Eval Loss | N/A |")
    lines.append("")

    # ── Loss 曲线 ──
    lines.append("### Train Loss")
    lines.append("")
    lines.append("```")
    lines.append(_ascii_loss_plot(train_losses, "loss"))
    lines.append("```")
    lines.append("")

    if eval_losses:
        lines.append("### Eval Loss")
        lines.append("")
        lines.append("```")
        lines.append(_ascii_loss_plot(eval_losses, "eval_loss"))
        lines.append("```")
        lines.append("")

    # ── Loss 数据表 ──
    if train_losses:
        lines.append("### Loss Log (first + last 5)")
        lines.append("")
        lines.append("| Step | Loss | LR | Time |")
        lines.append("|------|------|----|------|")
        show = train_losses[:5] + (train_losses[-5:] if len(train_losses) > 10 else [])
        for e in show:
            loss_str = f"{e.get('loss', 'N/A'):.6f}" if 'loss' in e else "N/A"
            lr_str = f"{e['lr']:.2e}" if 'lr' in e else "N/A"
            lines.append(f"| {e['step']} | {loss_str} | {lr_str} | {e['time_s']}s |")
        if len(train_losses) > 10:
            lines.append(f"| ... | ... | ... | ... |")
            lines.append(f"| *{len(train_losses)} total entries* | | | |")
        lines.append("")

    # ── GPU 统计 ──
    if gpu_samples:
        lines.append("## 💾 GPU Memory")
        lines.append("")
        g = r["gpu"]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Peak VRAM | {g['peak_allocated_gb']:.1f} GB |")
        lines.append(f"| Avg VRAM | {g['avg_allocated_gb']:.1f} GB |")
        lines.append(f"| GPU Model | {cfg['gpu']} ({cfg['gpu_memory_gb']:.0f} GB total) |")
        lines.append("")

    # ── 下一步 ──
    lines.append("## 🚀 Next Steps")
    lines.append("")
    lines.append("```bash")
    lines.append("# 推理生成提交文件")
    output_dir = getattr(config, "OUTPUT_DIR", "./output/ontology_lora")
    lines.append(f"python predict_ontology.py --model_path {output_dir}/final_model --num_sc 3")
    lines.append("")
    lines.append("# 本地评估")
    lines.append("python evaluate_local.py data/train_task_a.json submission.json")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
