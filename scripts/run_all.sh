#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# LLMs4OL 2026 — 一键训练 + 推理 + 评估
# ═══════════════════════════════════════════════════════════════
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }
step()  { echo -e "\n${BLUE}══ $* ══${NC}"; }

# ═══════════════════════════════════════════════════════════
step "1/6: Checking environment"
# ═══════════════════════════════════════════════════════════

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null \
    || error "CUDA not available! Activate conda env first: conda activate llms4ol2026"

gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
gpu_mem=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024**3))")
info "GPU: $gpu_name (${gpu_mem}GB)"

[ -f "data/train_task_a.json" ] || error "data/train_task_a.json not found!"
info "Training data: OK ($(du -h data/train_task_a.json | cut -f1))"

# ═══════════════════════════════════════════════════════════
step "2/6: Finding model"
# ═══════════════════════════════════════════════════════════

# 用 find 直接找 config.json — 不依赖 Python，不会被 print 污染
CONFIGS=($(find models/ -name config.json -type f -not -path '*/blobs/*' 2>/dev/null))

if [ ${#CONFIGS[@]} -eq 0 ]; then
    error "No model found in models/ directory!
  Download on a machine with internet:
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
  Then SCP to this server:
    scp -r models/Qwen2.5-7B-Instruct user@server:$(pwd)/models/"
fi

# 取第一个找到的模型目录
FOUND="$(dirname "${CONFIGS[0]}")"
FOUND="$(cd "$FOUND" && pwd)"

if [ ${#CONFIGS[@]} -gt 1 ]; then
    info "Multiple models found (${#CONFIGS[@]}), using first: $FOUND"
else
    info "Model: $FOUND"
fi

# ═══════════════════════════════════════════════════════════
step "3/6: Training"
# ═══════════════════════════════════════════════════════════

info "GPU: auto | Model: $FOUND"
START=$(date +%s)

python scripts/train.py --gpu auto --model "$FOUND"

END=$(date +%s)
DUR=$((END - START))
info "Training done in: $((DUR/3600))h $(((DUR%3600)/60))m $((DUR%60))s"

# ═══════════════════════════════════════════════════════════
step "4/6: Inference"
# ═══════════════════════════════════════════════════════════

LATEST=$(find output -name "final_model" -type d 2>/dev/null | sort | tail -1)
[ -z "$LATEST" ] && error "No trained model found in output/"

SUBMISSION="${SUBMISSION:-output/submission.json}"
info "Model checkpoint: $LATEST"

python scripts/predict.py \
    --model_path "$LATEST" \
    --base_model "$FOUND" \
    --output "$SUBMISSION" \
    --num_sc 3

info "Submission: $SUBMISSION ($(du -h "$SUBMISSION" 2>/dev/null | cut -f1))"

# ═══════════════════════════════════════════════════════════
step "5/6: Evaluation"
# ═══════════════════════════════════════════════════════════

[ -f "$SUBMISSION" ] && python scripts/evaluate.py data/train_task_a.json "$SUBMISSION"

# ═══════════════════════════════════════════════════════════
step "6/6: Report"
# ═══════════════════════════════════════════════════════════

echo ""
REPORT=$(find output -name "report.md" -type f 2>/dev/null | sort | tail -1)
if [ -n "$REPORT" ]; then
    info "Report: $REPORT"
    head -25 "$REPORT"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ ALL DONE                                            ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Submission : $SUBMISSION"
echo "║  Report     : $REPORT"
echo "║  Model      : $LATEST"
echo "╚══════════════════════════════════════════════════════════╝"
