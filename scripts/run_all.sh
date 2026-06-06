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
error() { echo -e "${RED}[✗]${NC} $*"; }
step()  { echo -e "\n${BLUE}══ $* ══${NC}"; }

# ═══════════════════════════════════════════════════════════
step "1/6: Checking environment"
# ═══════════════════════════════════════════════════════════

# CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
    error "CUDA not available! Activate conda env first: conda activate llms4ol2026"
    exit 1
}
gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
gpu_mem=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024**3))")
info "GPU: $gpu_name (${gpu_mem}GB)"

# 数据
[ -f "data/train_task_a.json" ] || { error "data/train_task_a.json not found!"; exit 1; }
info "Training data: OK ($(du -h data/train_task_a.json | cut -f1))"

# ═══════════════════════════════════════════════════════════
step "2/6: Checking model"
# ═══════════════════════════════════════════════════════════

# 自动检测 GPU → 推荐模型
if echo "$gpu_name" | grep -qi "a100"; then
    MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
elif echo "$gpu_name" | grep -qi "v100"; then
    MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
else
    MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
fi

info "Target model: $MODEL"

# 搜索已下载的模型
FOUND=$(python -c "
from src.config import find_local_model
import sys, os
# 尝试几个常见选项
for path in ['$MODEL',
             os.path.join('models', '${MODEL##*/}'),
             'models/hub/${MODEL##*/}/snapshots']:
    result = find_local_model(path)
    if os.path.isdir(result) and os.path.exists(os.path.join(result, 'config.json')):
        print(result)
        sys.exit(0)
print('NOT_FOUND')
" 2>/dev/null)

if [ "$FOUND" == "NOT_FOUND" ] || [ -z "$FOUND" ]; then
    echo ""
    error "Model not found! Search complete models/ directory."
    echo ""
    echo "  If you have the model somewhere in models/, try:"
    echo "    find models/ -name config.json -type f"
    echo ""
    echo "  Then pass the parent directory with --model:"
    echo "    python scripts/train.py --gpu auto --model <path>"
    exit 1
fi

info "Model found: $FOUND"

# 如果有多个模型，让用户选
if echo "$FOUND" | wc -l | grep -q "2"; then
    echo "  Multiple models found. Using the first one."
    FOUND=$(echo "$FOUND" | head -1)
fi

# ═══════════════════════════════════════════════════════════
step "3/6: Training"
# ═══════════════════════════════════════════════════════════

info "Starting training with model: $FOUND"
START=$(date +%s)

python scripts/train.py --gpu auto --model "$FOUND" "$@"

END=$(date +%s)
DUR=$((END - START))
info "Training done in: $((DUR/3600))h $(((DUR%3600)/60))m $((DUR%60))s"

# ═══════════════════════════════════════════════════════════
step "4/6: Inference"
# ═══════════════════════════════════════════════════════════

# 找最新模型
LATEST=$(find output -name "final_model" -type d 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    error "No trained model found in output/"
    exit 1
fi

SUBMISSION="${SUBMISSION:-output/submission.json}"
info "Model: $LATEST"
info "Output: $SUBMISSION"

python scripts/predict.py \
    --model_path "$LATEST" \
    --base_model "$MODEL" \
    --output "$SUBMISSION" \
    --num_sc 3

info "Submission saved: $SUBMISSION ($(du -h "$SUBMISSION" 2>/dev/null | cut -f1))"

# ═══════════════════════════════════════════════════════════
step "5/6: Evaluation (on training set)"
# ═══════════════════════════════════════════════════════════

if [ -f "data/train_task_a.json" ] && [ -f "$SUBMISSION" ]; then
    python scripts/evaluate.py data/train_task_a.json "$SUBMISSION"
fi

# ═══════════════════════════════════════════════════════════
step "6/6: Report"
# ═══════════════════════════════════════════════════════════

echo ""
REPORT=$(find output -name "report.md" -type f 2>/dev/null | sort | tail -1)
if [ -n "$REPORT" ]; then
    info "Training report: $REPORT"
    echo ""
    head -30 "$REPORT"
else
    warn "No report generated (check output/ for report.md)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ ALL DONE                                            ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Submission : $SUBMISSION"
echo "║  Report     : $REPORT"
echo "║  Model      : $LATEST"
echo "╚══════════════════════════════════════════════════════════╝"
