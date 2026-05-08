# Pre-training pipeline for the CyclingTransformer on a CPU machine.
#
# Run from the project root, e.g.:
#   .\ml\pretrain.ps1 -SmallTest               # ~10 min smoke test
#   .\ml\pretrain.ps1 -Athletes 15000 -Fast    # ~2-4h on Ryzen 5600
#   .\ml\pretrain.ps1 -Athletes 50000          # full pre-train, can run overnight+
#   .\ml\pretrain.ps1 -Athletes 15000 -Reuse   # skip re-generating data

param(
    [int]    $Athletes  = 50000,   # synthetic athletes to simulate
    [int]    $Epochs    = 0,       # 0 = pick a sensible default per mode
    [int]    $BatchSize = 64,
    [switch] $Fast,                # smaller transformer (d=64, 4 layers)
    [switch] $SmallTest,           # 500 athletes, 26 weeks, 20 epochs, fast model
    [switch] $Reuse                # don't regenerate parquet if it already exists
)

$ErrorActionPreference = "Stop"

# Both generator and trainer import `app.ml.*` — make backend importable.
$env:PYTHONPATH       = "$PWD\backend"
# Avoid Windows cp1252 UnicodeEncodeError on tqdm / arrow chars.
$env:PYTHONIOENCODING = "utf-8"
# Maximize CPU threads.
$env:OMP_NUM_THREADS  = [System.Environment]::ProcessorCount
$env:MKL_NUM_THREADS  = [System.Environment]::ProcessorCount

New-Item -ItemType Directory -Force -Path ".\backend\models" | Out-Null
New-Item -ItemType Directory -Force -Path ".\ml\data"        | Out-Null

# ── Pick mode-specific defaults ───────────────────────────────────────────────
if ($SmallTest) {
    $dataFile  = ".\ml\data\synthetic_small.parquet"
    $modelFile = ".\backend\models\cycling_coach_test.pt"
    if ($Epochs -eq 0) { $Epochs = 20 }
    $Athletes  = 500
    $weeks     = 26
    $useFast   = $true
    $modeLabel = "SMOKE TEST  (500 athletes, 26 weeks)"
} else {
    $dataFile  = ".\ml\data\synthetic.parquet"
    $modelFile = ".\backend\models\cycling_coach.pt"
    if ($Epochs -eq 0) { $Epochs = 100 }
    $weeks     = 0   # 0 = generator picks per-athlete random length (40-104)
    $useFast   = [bool]$Fast
    $modeLabel = "FULL PRE-TRAINING ($Athletes athletes)"
}

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== $modeLabel ===" -ForegroundColor Cyan
Write-Host "  Data file  : $dataFile"
Write-Host "  Model file : $modelFile"
Write-Host "  Epochs     : $Epochs"
Write-Host "  Batch size : $BatchSize"
Write-Host "  Model size : $(if ($useFast) {'FAST (d=64, 4 layers, ~221k params)'} else {'FULL (d=128, 6 layers, ~1M params)'})"
Write-Host "  CPU threads: $($env:OMP_NUM_THREADS)"
Write-Host ""

# ── Step 1: Generate synthetic data (or reuse) ───────────────────────────────
if ($Reuse -and (Test-Path $dataFile)) {
    $sizeMB = [math]::Round((Get-Item $dataFile).Length / 1MB, 1)
    Write-Host "[1/2] Reusing existing $dataFile ($sizeMB MB)" -ForegroundColor Yellow
} else {
    Write-Host "[1/2] Generating synthetic data..." -ForegroundColor Cyan
    $genArgs = @(
        "-m", "ml.training.generate_synthetic",
        "--athletes", $Athletes,
        "--output",   $dataFile
    )
    if ($weeks -gt 0) { $genArgs += @("--weeks", $weeks) }
    python @genArgs
    if ($LASTEXITCODE -ne 0) { throw "Synthetic data generation failed (exit $LASTEXITCODE)" }
}

# ── Step 2: Train ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/2] Training model..." -ForegroundColor Cyan

$trainArgs = @(
    "-m", "ml.training.train",
    "--data",       $dataFile,
    "--output",     $modelFile,
    "--epochs",     $Epochs,
    "--batch-size", $BatchSize
)
if ($useFast) { $trainArgs += "--fast" }

python @trainArgs
if ($LASTEXITCODE -ne 0) { throw "Training failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Model saved to: $modelFile"
Write-Host "Set ML_MODEL_PATH=$modelFile in your .env to use it."
