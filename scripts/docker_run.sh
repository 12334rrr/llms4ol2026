#!/bin/bash
# ═══════════════════════════════════════════════════════════
# LLMs4OL 2026 — Docker 快速启动脚本
# ═══════════════════════════════════════════════════════════

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

ACTION="${1:-help}"
shift || true

case "$ACTION" in
# ═══════════ 构建 ═══════════
build)
    echo "🐳 Building Docker image..."
    docker build -t llms4ol:latest .
    echo "✅ Build complete"
    ;;

# ═══════════ 训练 ═══════════
train)
    echo "🐳 Starting training..."
    docker run --rm --gpus all \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$PROJECT_DIR/output:/app/output" \
        -e HF_ENDPOINT=https://hf-mirror.com \
        llms4ol:latest \
        python scripts/train.py --gpu auto "$@"
    ;;

# ═══════════ 推理 ═══════════
predict)
    echo "🐳 Starting prediction..."
    docker run --rm --gpus all \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$PROJECT_DIR/output:/app/output" \
        -e HF_ENDPOINT=https://hf-mirror.com \
        llms4ol:latest \
        python scripts/predict.py \
            --model_path /app/output/ontology_lora/final_model \
            --test_path /app/data/test_task_a_input.json \
            --output /app/output/submission.json \
            "$@"
    ;;

# ═══════════ GPU 检查 ═══════════
gpu-check)
    echo "🐳 Checking GPU..."
    docker run --rm --gpus all \
        llms4ol:latest \
        python scripts/check_gpu.py
    ;;

# ═══════════ 交互 Shell ═══════════
shell)
    echo "🐳 Starting interactive shell..."
    docker run --rm -it --gpus all \
        -v "$PROJECT_DIR/data:/app/data" \
        -v "$PROJECT_DIR/models:/app/models" \
        -v "$PROJECT_DIR/output:/app/output" \
        -e HF_ENDPOINT=https://hf-mirror.com \
        llms4ol:latest \
        /bin/bash
    ;;

# ═══════════ Docker Compose 方式 ═══════════
compose-train)
    docker-compose build
    docker-compose up train
    ;;
compose-predict)
    docker-compose up predict
    ;;

# ═══════════ 帮助 ═══════════
*)
    echo "LLMs4OL 2026 — Docker Deployment"
    echo ""
    echo "Usage: bash scripts/docker_run.sh <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  build          Build Docker image"
    echo "  train          Start training (GPU auto-detect)"
    echo "  predict        Generate submission file"
    echo "  gpu-check      Check GPU availability"
    echo "  shell          Interactive bash shell in container"
    echo "  compose-train  Train via docker-compose"
    echo "  compose-predict Predict via docker-compose"
    echo ""
    echo "Examples:"
    echo "  bash scripts/docker_run.sh build"
    echo "  bash scripts/docker_run.sh train --epochs 5 --lr 1e-4"
    echo "  bash scripts/docker_run.sh predict --num_sc 5"
    echo "  bash scripts/docker_run.sh shell"
    ;;
esac
