#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# LLMs4OL 2026 — 创建私有 GitHub 仓库并推送代码
# ═══════════════════════════════════════════════════════════════
#
# 前置条件: 安装 GitHub CLI
#   Linux:   sudo apt install gh   (或 brew install gh)
#   Windows: winget install --id GitHub.cli
#   登录:    gh auth login
#
# 用法:
#   bash scripts/init_github.sh                     # 交互式
#   bash scripts/init_github.sh -n my-repo -d "desc" # 指定参数
# ═══════════════════════════════════════════════════════════════

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ── 参数 ──
REPO_NAME=""
REPO_DESC="LLMs4OL 2026 — End-to-End Ontology Learning"
PRIVATE=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)     REPO_NAME="$2"; shift 2 ;;
        -d|--desc)     REPO_DESC="$2"; shift 2 ;;
        --public)      PRIVATE=false; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/init_github.sh [options]"
            echo ""
            echo "Options:"
            echo "  -n, --name   REPO_NAME    Repository name (default: llms4ol2026)"
            echo "  -d, --desc   DESCRIPTION  Repository description"
            echo "  --public                  Make repository public (default: private)"
            echo "  --dry-run                 Show what would be done without doing it"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[ -z "$REPO_NAME" ] && REPO_NAME="llms4ol2026"

# ── 颜色 ──
G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${G}[INFO]${NC}  $*"; }
warn()  { echo -e "${Y}[WARN]${NC}  $*"; }
step()  { echo -e "\n${B}════════════════════════════════════════${NC}"; echo -e "${B}  $*${NC}"; echo -e "${B}════════════════════════════════════════${NC}"; }

# ═══════════════════════════════════════════════════════════
step "1/5: Checking prerequisites"
# ═══════════════════════════════════════════════════════════

if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) not found!"
    echo ""
    echo "Install it first:"
    echo "  Linux:   sudo apt install gh"
    echo "  macOS:   brew install gh"
    echo "  Windows: winget install --id GitHub.cli"
    echo ""
    echo "Then login:  gh auth login"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    warn "Not logged in to GitHub. Run: gh auth login"
    exit 1
fi

info "GitHub CLI: OK (logged in as $(gh api user --jq '.login'))"

# ═══════════════════════════════════════════════════════════
step "2/5: Initializing Git repo"
# ═══════════════════════════════════════════════════════════

if [ ! -d ".git" ]; then
    info "Initializing Git repository..."
    git init
    git branch -M main
else
    info "Git repository already exists"
    # Ensure we're on main branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        git checkout -b main 2>/dev/null || git checkout main 2>/dev/null || true
    fi
fi

# ═══════════════════════════════════════════════════════════
step "3/5: Creating .gitignore"
# ═══════════════════════════════════════════════════════════

if [ ! -f ".gitignore" ]; then
    info "Creating .gitignore..."
    cat > .gitignore << 'EOF'
models/
output/
__pycache__/
*.py[cod]
*.egg-info/
.idea/
.vscode/
.DS_Store
Thumbs.db
submission*.json
*.safetensors
*.pt
*.pth
*.bin
*.ckpt
EOF
    info ".gitignore created"
fi

# ═══════════════════════════════════════════════════════════
step "4/5: Creating GitHub repository"
# ═══════════════════════════════════════════════════════════

echo ""
echo "  Repository: $REPO_NAME"
echo "  Visibility: $([ "$PRIVATE" = true ] && echo '🔒 Private' || echo '🌐 Public')"
echo "  Description: $REPO_DESC"
echo ""

EXISTING=$(gh repo list --json name --jq '.[].name' 2>/dev/null | grep "^${REPO_NAME}$" || true)

if [ -n "$EXISTING" ]; then
    warn "Repository '$REPO_NAME' already exists on GitHub"
    read -p "  Use existing repo? [Y/n] " -n 1 -r; echo
    if [[ $REPLY =~ ^[Nn] ]]; then
        REPO_NAME="${REPO_NAME}-$(date +%Y%m%d)"
        info "Using new name: $REPO_NAME"
    fi
fi

if [ "$DRY_RUN" = true ]; then
    info "[DRY RUN] Would create: $REPO_NAME (private=$PRIVATE)"
else
    VISIBILITY="--private"
    [ "$PRIVATE" = false ] && VISIBILITY="--public"

    gh repo create "$REPO_NAME" $VISIBILITY \
        --description "$REPO_DESC" \
        --source . \
        --remote origin \
        --push 2>&1 || {
        # 如果 --push 失败，手动 push
        info "Creating repo without push, will push manually..."
        gh repo create "$REPO_NAME" $VISIBILITY --description "$REPO_DESC" --source . --remote origin
    }

    info "Repository created: https://github.com/$(gh api user --jq '.login')/$REPO_NAME"
fi

# ═══════════════════════════════════════════════════════════
step "5/5: Pushing code"
# ═══════════════════════════════════════════════════════════

if [ "$DRY_RUN" = true ]; then
    info "[DRY RUN] Would push to GitHub"
    info "[DRY RUN] Files to be committed:"
    git status --short | head -20
else
    # 添加所有文件
    git add .
    git add .gitignore -f 2>/dev/null || true

    # 检查是否有更改
    if git diff --cached --quiet 2>/dev/null; then
        info "No changes to commit — already up to date"
    else
        git commit -m "Initial commit: LLMs4OL 2026 ontology learning pipeline

- src/: Core modules (config, data, model, postprocess, report)
- scripts/: Training, inference, evaluation, deployment
- configs/: GPU presets (A100, V100, T4, RTX3090)
- Dockerfile + docker-compose for containerized deployment" 2>/dev/null || {
            info "Using default commit message..."
            git commit -m "Initial commit" 2>/dev/null || true
        }
        info "Committed changes"
    fi

    # Push
    git push -u origin main 2>&1 || git push -u origin master 2>&1 || {
        warn "Push failed. Trying to set remote..."
        USERNAME=$(gh api user --jq '.login')
        git remote add origin "https://github.com/${USERNAME}/${REPO_NAME}.git" 2>/dev/null || true
        git push -u origin main 2>&1 || git push -u origin master 2>&1
    }
fi

# ═══════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ✅  Repository Ready!                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
USERNAME=$(gh api user --jq '.login' 2>/dev/null || echo "YOUR_USER")
echo "║  URL: https://github.com/${USERNAME}/${REPO_NAME}"
echo "║                                                          ║"
echo "║  Clone on another server:                               ║"
echo "║    git clone https://github.com/${USERNAME}/${REPO_NAME}.git"
echo "╚══════════════════════════════════════════════════════════╝"
