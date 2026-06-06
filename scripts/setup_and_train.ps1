# ═══════════════════════════════════════════════════════════════
# LLMs4OL 2026 — 一键部署 + 训练脚本 (Windows PowerShell)
# ═══════════════════════════════════════════════════════════════
#
# 用法:
#   .\scripts\setup_and_train.ps1
#   .\scripts\setup_and_train.ps1 -Gpu auto -Epochs 5
#   .\scripts\setup_and_train.ps1 -SkipInstall
#   .\scripts\setup_and_train.ps1 -InferenceOnly
# ═══════════════════════════════════════════════════════════════

param(
    [string]$Gpu = "auto",
    [int]$Epochs = 3,
    [string]$BatchSize = "",
    [string]$LR = "",
    [switch]$SkipInstall = $false,
    [switch]$InferenceOnly = $false
)

$ErrorActionPreference = "Stop"
$ENV_NAME = "llms4ol2026"
$PROJECT_DIR = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $PROJECT_DIR

function Write-Step($msg) {
    Write-Host ""
    Write-Host "════════════════════════════════════════" -ForegroundColor Blue
    Write-Host "  $msg" -ForegroundColor Blue
    Write-Host "════════════════════════════════════════" -ForegroundColor Blue
}

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Error-Exit($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ═══════════════════════════════════════════════════════════
# STEP 1-3: 创建环境 + 安装依赖
# ═══════════════════════════════════════════════════════════
if (-not $SkipInstall) {
    Write-Step "STEP 1: Creating conda environment: $ENV_NAME"

    $condaExists = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaExists) {
        Write-Error-Exit "conda not found!"
    }

    $existingEnv = conda env list | Select-String "^${ENV_NAME} "
    if ($existingEnv) {
        Write-Warn "Environment '$ENV_NAME' exists, removing..."
        conda env remove -n $ENV_NAME -y
    }

    Write-Info "Creating conda env with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y

    Write-Step "STEP 2: Installing PyTorch with CUDA"

    conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    Write-Step "STEP 3: Installing project dependencies"
    conda run -n $ENV_NAME pip install -r requirements.txt
    conda run -n $ENV_NAME pip install accelerate

    Write-Info "Verifying CUDA..."
    conda run -n $ENV_NAME python -c @"
import torch
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE!'
print(f'CUDA: {torch.version.cuda}')
print(f'GPU:  {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"@
    Write-Info "Environment setup complete!"
}
else {
    Write-Info "Skipping environment setup (--SkipInstall)"
}

# ═══════════════════════════════════════════════════════════
# STEP 4: GPU 诊断
# ═══════════════════════════════════════════════════════════
Write-Step "STEP 4: GPU Diagnostics"
conda run -n $ENV_NAME python scripts/check_gpu.py

# ═══════════════════════════════════════════════════════════
# STEP 5: 训练
# ═══════════════════════════════════════════════════════════
if (-not $InferenceOnly) {
    Write-Step "STEP 5: Training (GPU=$Gpu, Epochs=$Epochs)"

    $trainCmd = "python scripts/train.py --gpu $Gpu --epochs $Epochs"
    if ($BatchSize) { $trainCmd += " --batch_size $BatchSize" }
    if ($LR)        { $trainCmd += " --lr $LR" }

    Write-Info "Command: $trainCmd"
    $startTime = Get-Date

    conda run -n $ENV_NAME $trainCmd

    $duration = (Get-Date) - $startTime
    Write-Info "Training completed in: $($duration.ToString('hh\h mm\m ss\s'))"
}
else {
    Write-Info "Skipping training (--InferenceOnly)"
}

# ═══════════════════════════════════════════════════════════
# STEP 6: 推理
# ═══════════════════════════════════════════════════════════
Write-Step "STEP 6: Inference"

$modelPath = Get-ChildItem -Path "output\*\final_model" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $modelPath) {
    Write-Warn "No trained model found, using base model (zero-shot)..."
    conda run -n $ENV_NAME python scripts/predict.py --no_lora --output output/submission_zeroshot.json
    $SUBMISSION = "output/submission_zeroshot.json"
}
else {
    Write-Info "Using model: $modelPath"
    conda run -n $ENV_NAME python scripts/predict.py --model_path $modelPath --output output/submission.json --num_sc 3
    $SUBMISSION = "output/submission.json"
}

# ═══════════════════════════════════════════════════════════
# STEP 7: 评估
# ═══════════════════════════════════════════════════════════
Write-Step "STEP 7: Local Evaluation"

if ((Test-Path $SUBMISSION) -and (Test-Path "data/train_task_a.json")) {
    conda run -n $ENV_NAME python scripts/evaluate.py data/train_task_a.json $SUBMISSION
}
else {
    Write-Warn "Skipping evaluation"
}

# ═══════════════════════════════════════════════════════════
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              ✓  ALL DONE!                               ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  Environment : $ENV_NAME" -ForegroundColor Green
Write-Host "║  Submission  : $SUBMISSION" -ForegroundColor Green
Write-Host "║  Reports     : output\*\report.md" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Green
