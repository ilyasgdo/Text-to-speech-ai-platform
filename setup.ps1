# Windows Setup Script for Qwen3-TTS Studio
$ErrorActionPreference = "Stop"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "🎵 Qwen3-TTS Studio - Installation (Windows)" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Check for Python
Write-Host "`n📦 Checking Python..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Please install Python 3.10+ and add it to PATH." -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "`n🔧 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n🔌 Activating virtual environment..." -ForegroundColor Yellow
$venvPath = ".venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    . $venvPath
} else {
    Write-Host "❌ Could not find activation script at $venvPath" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n📥 Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip

# Install PyTorch with CUDA support
Write-Host "   Installing PyTorch with CUDA 12.4 support..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "⚠️ requirements.txt not found!" -ForegroundColor Yellow
}

# Check for Ollama
Write-Host "`n🦙 Checking Ollama..." -ForegroundColor Yellow
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "✅ Ollama is installed" -ForegroundColor Green
    Write-Host "💡 To use Ollama features, run in a separate terminal: ollama serve" -ForegroundColor Yellow
} else {
    Write-Host "⚠️ Ollama is not installed (optional)" -ForegroundColor Yellow
    Write-Host "   Download from https://ollama.com/"
}

Write-Host "`n==============================================" -ForegroundColor Green
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

Write-Host "`n🚀 Launching application..." -ForegroundColor Yellow
python qwen_tts_studio.py
