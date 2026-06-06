#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 模型下载脚本 — 在有网络的机器上运行
# ═══════════════════════════════════════════════════════════════
#
# 用法:
#   bash scripts/download_models.sh              # 下载全部模型
#   bash scripts/download_models.sh --7b          # 只下载 7B
#   bash scripts/download_models.sh --14b         # 只下载 14B
#   bash scripts/download_models.sh --all         # 全部 (7B + 14B + 3B)
# ═══════════════════════════════════════════════════════════════

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

G='\033[0;32m'; Y='\033[1;33m'; NC='\033[0m'
info() { echo -e "${G}[INFO]${NC}  $*"; }
warn() { echo -e "${Y}[WARN]${NC}  $*"; }

# ── 参数 ──
DL_7B=false; DL_14B=false; DL_3B=false
if [ $# -eq 0 ]; then DL_7B=true; DL_14B=true; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --7b)  DL_7B=true; shift ;;
        --14b) DL_14B=true; shift ;;
        --3b)  DL_3B=true; shift ;;
        --all) DL_7B=true; DL_14B=true; DL_3B=true; shift ;;
        *) shift ;;
    esac
done

# ── 检查 huggingface-cli ──
if ! python -c "import huggingface_hub" 2>/dev/null; then
    info "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# ── 镜像 ──
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
info "HF endpoint: $HF_ENDPOINT"

mkdir -p models

# ── 下载函数 ──
download_model() {
    local name="$1"
    local short="$2"
    local dir="models/$short"

    if [ -f "$dir/config.json" ]; then
        info "Already exists: $short ($dir)"
        return
    fi

    info "Downloading $name → $dir ..."
    huggingface-cli download "$name" --local-dir "$dir"
    info "Done: $short"
}

# ── 执行 ──
echo ""
info "Models will be saved to: $(pwd)/models/"
echo ""

[ "$DL_7B"  = true ] && download_model "Qwen/Qwen2.5-7B-Instruct"   "Qwen2.5-7B-Instruct"
[ "$DL_14B" = true ] && download_model "Qwen/Qwen2.5-14B-Instruct"  "Qwen2.5-14B-Instruct"
[ "$DL_3B"  = true ] && download_model "Qwen/Qwen2.5-3B-Instruct"   "Qwen2.5-3B-Instruct"

# ── 总结 ──
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ Models downloaded to: models/                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
ls -lh models/*/config.json 2>/dev/null | awk '{print "║  " $NF}'
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  To copy to server:                                     ║"
echo "║    scp -r models/ user@server:~/llms4ol2026/            ║"
echo "╚══════════════════════════════════════════════════════════╝"
