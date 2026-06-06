#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# LLMs4OL 2026 — 一键部署 + 训练脚本 (Conda 环境)
# ═══════════════════════════════════════════════════════════════
#
# 使用场景: 只有 conda 的裸服务器，从零开始到训练完成
#
# 用法:
#   bash scripts/setup_and_train.sh              # 默认: 自动检测GPU, 训练3 epoch
#   bash scripts/setup_and_train.sh --gpu v100   # 指定GPU类型
#   bash scripts/setup_and_train.sh --epochs 5 --lr 1e-4
#   bash scripts/setup_and_train.sh --skip-install  # 跳过环境安装
#   bash scripts/setup_and_train.sh --inference-only # 只做推理
# ═══════════════════════════════════════════════════════════════

set -e

# ── 配置 ──
ENV_NAME="llms4ol2026"
PYTHON_VER="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── 默认参数 ──
GPU="auto"
EPOCHS=3
BATCH_SIZE=""
LR=""
SKIP_INSTALL=false
INFERENCE_ONLY=false
TRAIN_ARGS=""

# ── 颜色 ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BLUE}════════════════════════════════════════${NC}"; echo -e "${BLUE}  $*${NC}"; echo -e "${BLUE}════════════════════════════════════════${NC}"; }

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --lr)          LR="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        --inference-only) INFERENCE_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/setup_and_train.sh [options]"
            echo ""
            echo "Options:"
            echo "  --gpu auto|a100|v100|t4    GPU type (default: auto-detect)"
            echo "  --epochs N                  Training epochs (default: 3)"
            echo "  --batch_size N              Batch size (default: auto)"
            echo "  --lr 2e-4                   Learning rate (default: auto)"
            echo "  --skip-install              Skip conda env creation"
            echo "  --inference-only            Only run inference"
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

cd "$PROJECT_DIR"

# ═══════════════════════════════════════════════════════════
# STEP 1: 创建 conda 环境 + 安装依赖
# ═══════════════════════════════════════════════════════════
if [ "$SKIP_INSTALL" = false ]; then
    step "STEP 1: Creating conda environment: $ENV_NAME"

    # 检查 conda
    if ! command -v conda &> /dev/null; then
        error "conda not found! Please install Miniconda first:\n  https://docs.conda.io/en/latest/miniconda.html"
    fi

    # 创建环境 (如果已存在则重建)
    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Environment '$ENV_NAME' exists, removing..."
        conda env remove -n "$ENV_NAME" -y
    fi

    info "Creating conda env with Python $PYTHON_VER..."
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

    # 获取 conda 环境的 pip 路径
    CONDA_PIP="$(conda run -n "$ENV_NAME" which pip)"
    info "Pip path: $CONDA_PIP"

    # ── 安装 PyTorch (CUDA 版) ──
    step "STEP 2: Installing PyTorch with CUDA"

    # 检测 CUDA 版本
    CUDA_VER=""
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2 | tr -d '.')
        info "Detected CUDA version from nvidia-smi: $CUDA_VER"
    fi

    if [ -z "$CUDA_VER" ]; then
        CUDA_VER="121"  # 默认 CUDA 12.1
        warn "Cannot detect CUDA version, using default: 12.1"
    fi

    # 根据 CUDA 版本选择 PyTorch 索引
    case "$CUDA_VER" in
        124|125) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        121|122) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        118)     TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        *)       TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    esac

    info "Installing PyTorch from: $TORCH_INDEX"
    conda run -n "$ENV_NAME" pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

    # ── 安装项目依赖 ──
    step "STEP 3: Installing project dependencies"

    conda run -n "$ENV_NAME" pip install -r requirements.txt
    conda run -n "$ENV_NAME" pip install accelerate

    # 验证 CUDA
    info "Verifying CUDA..."
    conda run -n "$ENV_NAME" python -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE!'
print(f'CUDA: {torch.version.cuda}')
print(f'GPU:  {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
    info "Environment setup complete!"
else
    info "Skipping environment setup (--skip-install)"
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        error "Environment '$ENV_NAME' not found and --skip-install is set!"
    fi
fi

# ═══════════════════════════════════════════════════════════
# STEP 4: GPU 诊断
# ═══════════════════════════════════════════════════════════
step "STEP 4: GPU Diagnostics"

conda run -n "$ENV_NAME" python scripts/check_gpu.py

# ═══════════════════════════════════════════════════════════
# STEP 5: 训练
# ═══════════════════════════════════════════════════════════
if [ "$INFERENCE_ONLY" = false ]; then
    step "STEP 5: Training ($GPU GPU, $EPOCHS epochs)"

    TRAIN_CMD="python scripts/train.py --gpu $GPU --epochs $EPOCHS"
    [ -n "$BATCH_SIZE" ] && TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
    [ -n "$LR" ]         && TRAIN_CMD="$TRAIN_CMD --lr $LR"

    info "Command: $TRAIN_CMD"
    echo ""

    # 记录开始时间
    START_TIME=$(date +%s)

    conda run -n "$ENV_NAME" $TRAIN_CMD

    # 计算耗时
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    info "Training completed in: ${HOURS}h ${MINUTES}m ${SECONDS}s"
else
    info "Skipping training (--inference-only)"
fi

# ═══════════════════════════════════════════════════════════
# STEP 6: 推理 + 生成提交文件
# ═══════════════════════════════════════════════════════════
step "STEP 6: Inference"

# 查找最新的模型
MODEL_PATH=""
for d in output/*/final_model; do
    if [ -d "$d" ]; then
        MODEL_PATH="$d"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    warn "No trained model found! Trying to predict with base model..."
    conda run -n "$ENV_NAME" python scripts/predict.py \
        --no_lora \
        --output output/submission_zeroshot.json
else
    info "Using model: $MODEL_PATH"
    conda run -n "$ENV_NAME" python scripts/predict.py \
        --model_path "$MODEL_PATH" \
        --output output/submission.json \
        --num_sc 3
fi

# ═══════════════════════════════════════════════════════════
# STEP 7: 本地评估 (如果有标注数据)
# ═══════════════════════════════════════════════════════════
step "STEP 7: Local Evaluation"

SUBMISSION="output/submission.json"
[ ! -f "$SUBMISSION" ] && SUBMISSION="output/submission_zeroshot.json"

if [ -f "$SUBMISSION" ] && [ -f "data/train_task_a.json" ]; then
    conda run -n "$ENV_NAME" python scripts/evaluate.py data/train_task_a.json "$SUBMISSION"
else
    warn "Skipping evaluation (submission or gold file not found)"
fi

# ═══════════════════════════════════════════════════════════
# 完成
# ═══════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ✅  ALL DONE!                               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Environment : $ENV_NAME"
echo "║  Model       : $MODEL_PATH"
echo "║  Submission  : $SUBMISSION"
echo "║  Reports     : output/*/report.md"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Activate env for manual use:                           ║"
echo "║    conda activate $ENV_NAME"
echo "║    python scripts/predict.py --help                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
