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
    step "STEP 1: Setting up conda environment: $ENV_NAME"

    # 检查 conda
    if ! command -v conda &> /dev/null; then
        error "conda not found! Please install Miniconda first:\n  https://docs.conda.io/en/latest/miniconda.html"
    fi

    # 环境不存在则创建，存在则直接使用
    if conda env list | grep -q "^${ENV_NAME} "; then
        info "Environment '$ENV_NAME' already exists, reusing..."
    else
        info "Creating conda env with Python $PYTHON_VER..."
        conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
    fi

    # 获取 conda 环境的 pip 路径
    CONDA_PIP="$(conda run -n "$ENV_NAME" which pip)"
    info "Pip path: $CONDA_PIP"

    # ── 安装 PyTorch (CUDA 版) ──
    step "STEP 2: Installing PyTorch with CUDA"

    # ── 检测 GPU Driver 版本 (不是 CUDA toolkit 版本!) ──
    DRIVER_VER=""
    CUDA_MAX=""
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1)
        CUDA_MAX=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2 | tr -d '.')
        info "NVIDIA Driver: $DRIVER_VER, Max CUDA: $CUDA_MAX"
    fi

    # ── 选择合适的 PyTorch CUDA 版本 ──
    # 关键: PyTorch 的 CUDA 版本不能超过 Driver 支持的最大版本
    # Driver >= 545 → cu124, Driver >= 525 → cu121, 否则 cu118
    if [ -n "$DRIVER_VER" ] && [ "$DRIVER_VER" -ge 545 ] 2>/dev/null; then
        CUDA_TAG="cu124"
    elif [ -n "$DRIVER_VER" ] && [ "$DRIVER_VER" -ge 525 ] 2>/dev/null; then
        CUDA_TAG="cu121"
    else
        CUDA_TAG="cu118"
        info "Driver too old, using PyTorch CUDA 11.8 (driver-compatible)"
    fi
    info "PyTorch CUDA tag: $CUDA_TAG"

    # ── PyTorch 国内安装策略 ──
    # 方案1: conda + 清华镜像 (最稳)
    # 方案2: pip + 上交镜像
    # 方案3: pip + 官方源 (慢但一定能用)
    # 方案4: conda + 官方源

    info "Installing PyTorch $CUDA_TAG via conda (Tsinghua mirror)..."
    echo ""

    # conda CUDA 版本映射
    case "$CUDA_TAG" in
        cu124) CONDA_CUDA="12.4" ;;
        cu121) CONDA_CUDA="12.1" ;;
        cu118) CONDA_CUDA="11.8" ;;
    esac

    # 尝试 conda + 清华源
    conda run -n "$ENV_NAME" conda install pytorch \
        "pytorch-cuda=$CONDA_CUDA" torchvision torchaudio \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
        -c nvidia -y 2>&1 | tail -3
    CONDA_EXIT=${PIPESTATUS[0]}

    # 验证是否安装成功
    conda run -n "$ENV_NAME" python -c "import torch; print('torch', torch.__version__)" 2>/dev/null
    PYTORCH_OK=$?

    if [ "$PYTORCH_OK" != "0" ]; then
        warn "conda mirror failed, trying pip + SJTU mirror..."
        conda run -n "$ENV_NAME" pip install torch torchvision torchaudio \
            --index-url "https://mirror.sjtu.edu.cn/pytorch-wheels/$CUDA_TAG/" \
            --timeout 300 2>&1 | tail -3
        PYTORCH_OK=$?
    fi

    if [ "$PYTORCH_OK" != "0" ]; then
        warn "SJTU mirror failed, trying official PyTorch index (slower)..."
        conda run -n "$ENV_NAME" pip install torch torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
            --timeout 600 2>&1 | tail -3
        PYTORCH_OK=$?
    fi

    if [ "$PYTORCH_OK" != "0" ]; then
        warn "pip failed, trying conda official..."
        conda run -n "$ENV_NAME" conda install pytorch \
            "pytorch-cuda=$CONDA_CUDA" torchvision torchaudio \
            -c pytorch -c nvidia -y 2>&1 | tail -3
        PYTORCH_OK=$?
    fi

    # 最终验证
    if ! conda run -n "$ENV_NAME" python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        error "PyTorch CUDA installation FAILED!\n  Try manually:\n  conda activate $ENV_NAME\n  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_TAG"
    fi

    info "PyTorch CUDA: OK"

    # ── pip 配置国内源 ──
    step "STEP 3: Installing project dependencies"

    PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
    info "Using pip mirror: $PIP_MIRROR"

    conda run -n "$ENV_NAME" pip install -i "$PIP_MIRROR" --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt
    conda run -n "$ENV_NAME" pip install -i "$PIP_MIRROR" --trusted-host pypi.tuna.tsinghua.edu.cn accelerate

    # 验证 CUDA
    info "Verifying CUDA..."
    conda run -n "$ENV_NAME" python -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE!'
print(f'CUDA: {torch.version.cuda}')
print(f'GPU:  {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
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
