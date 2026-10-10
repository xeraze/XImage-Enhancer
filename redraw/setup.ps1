# One-shot setup for the SUPIR redraw service (Windows).
# Clones SUPIR, creates a venv, installs deps, applies patches, downloads models.
# Safe to re-run; each step is skipped if already done.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv"
$supir = Join-Path $root "SUPIR"
$python = "3.11"

Write-Host "==> Checking Python $python..."
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { Write-Host "Install Python $python from https://python.org (check 'py launcher')"; exit 1 }

if (-not (Test-Path $supir)) {
    Write-Host "==> Cloning SUPIR..."
    git clone --depth 1 https://github.com/Fanghua-Yu/SUPIR $supir
} else {
    Write-Host "==> SUPIR already cloned"
}

if (-not (Test-Path $venv)) {
    Write-Host "==> Creating venv..."
    py -$python -m venv $venv
} else {
    Write-Host "==> venv already exists"
}

$pip = Join-Path $venv "Scripts\pip.exe"
$pyExe = Join-Path $venv "Scripts\python.exe"

Write-Host "==> Installing requirements..."
& $pip install --upgrade pip
& $pip install -r (Join-Path $root "requirements.txt")
# PyTorch from the default index is CPU-only on Windows — reinstall with CUDA.
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "==> Downloading models (~20 GB, grab a coffee)..."
& $pyExe (Join-Path $root "download_models.py")

Write-Host "==> Applying compatibility patches..."
& $pyExe (Join-Path $root "apply_patches.py")

Write-Host ""
Write-Host "Setup complete. Start the service with:"
Write-Host "  run.bat"
