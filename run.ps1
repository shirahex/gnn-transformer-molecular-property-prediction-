# run.ps1 - Windows Startup Script for Molecular AI Project
# Run this in VS Code PowerShell terminal

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Molecular AI Project - Windows Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Detect Python command
$PYTHON = "python"
try {
    $pyVersion = & python --version 2>$null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    try {
        $pyVersion = & python3 --version 2>$null
        $PYTHON = "python3"
    } catch {
        Write-Host "ERROR: Python not found. Install Python 3.9+ from python.org" -ForegroundColor Red
        exit 1
    }
}
Write-Host "Found: $PYTHON" -ForegroundColor Green

# Check Node.js
try {
    $nodeVersion = & node --version
    Write-Host "Found Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Node.js not found. Install Node.js 18+ from nodejs.org" -ForegroundColor Red
    exit 1
}

# Get project root (where this script is)
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$BACKEND_DIR = Join-Path $PROJECT_ROOT "backend"
$FRONTEND_DIR = Join-Path $PROJECT_ROOT "frontend"
$DATA_DIR = Join-Path $PROJECT_ROOT "data"
$MODEL_DIR = Join-Path $PROJECT_ROOT "models"
$OUTPUTS_DIR = Join-Path $PROJECT_ROOT "outputs"

# Create directories if missing
@($DATA_DIR, $MODEL_DIR, $OUTPUTS_DIR) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null }
}

Write-Host ""
Write-Host "Project root: $PROJECT_ROOT" -ForegroundColor Yellow
Write-Host ""

# =========================================
# BACKEND SETUP
# =========================================
Write-Host "--- Setting up Backend ---" -ForegroundColor Cyan

Set-Location $BACKEND_DIR

# Create virtual environment if missing
$VENV_PATH = Join-Path $BACKEND_DIR "venv"
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & $PYTHON -m venv venv
}

# Activate virtual environment
$VENV_PYTHON = Join-Path $VENV_PATH "Scripts\python.exe"
$VENV_PIP = Join-Path $VENV_PATH "Scripts\pip.exe"

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: Virtual environment creation failed." -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $VENV_PYTHON -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Yellow
& $VENV_PIP install -r requirements.txt

# Check if PyTorch Geometric installed correctly
try {
    $null = & $VENV_PYTHON -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"
    Write-Host "PyTorch Geometric OK" -ForegroundColor Green
} catch {
    Write-Host "WARNING: PyTorch Geometric not found. Installing separately..." -ForegroundColor Yellow

    # Detect CUDA version
    $CUDA_VERSION = "cpu"
    try {
        $nvidia = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
        if ($nvidia) {
            Write-Host "NVIDIA GPU detected. Installing CUDA-enabled PyTorch..." -ForegroundColor Yellow
            # Try CUDA 11.8 (most compatible)
            & $VENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            & $VENV_PIP install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
        }
    } catch {
        Write-Host "No NVIDIA GPU detected. Installing CPU-only version..." -ForegroundColor Yellow
        & $VENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        & $VENV_PIP install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    }
}

# Check RDKit
try {
    $null = & $VENV_PYTHON -c "import rdkit; print(f'RDKit version: {rdkit.__version__}')"
    Write-Host "RDKit OK" -ForegroundColor Green
} catch {
    Write-Host "WARNING: RDKit not found. Installing..." -ForegroundColor Yellow
    & $VENV_PIP install rdkit
}

Write-Host ""
Write-Host "Backend setup complete!" -ForegroundColor Green
Write-Host ""

# =========================================
# FRONTEND SETUP
# =========================================
Write-Host "--- Setting up Frontend ---" -ForegroundColor Cyan
Set-Location $FRONTEND_DIR

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing npm packages..." -ForegroundColor Yellow
    & npm install
} else {
    Write-Host "node_modules already exists. Skipping npm install." -ForegroundColor Yellow
}

Write-Host "Frontend setup complete!" -ForegroundColor Green
Write-Host ""

# =========================================
# START SERVERS
# =========================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Servers..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start backend in a new window
Write-Host "Starting Backend (FastAPI) on http://localhost:8000 ..." -ForegroundColor Green
$BackendCmd = "cd `"$BACKEND_DIR`"; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd

Start-Sleep -Seconds 3

# Start frontend in a new window
Write-Host "Starting Frontend (React) on http://localhost:3000 ..." -ForegroundColor Green
$FrontendCmd = "cd `"$FRONTEND_DIR`"; npm start"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Both servers are starting!" -ForegroundColor Green
Write-Host "  Backend: http://localhost:8000" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit this window (servers will keep running)..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
