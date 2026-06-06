# ═══════════════════════════════════════════════════════════
# LLMs4OL 2026 — Docker 镜像
# ═══════════════════════════════════════════════════════════
#
# 构建:  docker build -t llms4ol:latest .
# 运行:  docker-compose up
#
# 基于 pytorch/pytorch 官方镜像 (自带 CUDA + PyTorch)
# ═══════════════════════════════════════════════════════════

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

LABEL maintainer="LLMs4OL 2026 Team"
LABEL description="End-to-End Ontology Learning — Flagship Task"

# 禁用交互式安装 (避免 tzdata 等卡住)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ═══════════════════════════════════════════════════════════
# 系统依赖
# ═══════════════════════════════════════════════════════════
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ═══════════════════════════════════════════════════════════
# Python 依赖
# ═══════════════════════════════════════════════════════════
COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir accelerate

# ═══════════════════════════════════════════════════════════
# 项目代码 (模型/数据/输出用 volume 挂载, 不进镜像)
# ═══════════════════════════════════════════════════════════
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
# 向后兼容入口 (thin wrappers)
COPY train_ontology.py predict_ontology.py evaluate_local.py main.py /app/

# 目录结构
RUN mkdir -p /app/data /app/models /app/output

# ═══════════════════════════════════════════════════════════
# 国内镜像加速 (可选: 构建时设置)
# ═══════════════════════════════════════════════════════════
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/app/models
ENV HF_HUB_CACHE=/app/models/hub
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ═══════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════
ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--gpu", "auto"]
