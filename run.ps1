# PowerShell Run Script for Windows
param (
    [switch]$Api
)

$ErrorActionPreference = "Stop"

$venvPath = Join-Path $PSScriptRoot "env"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvActivate = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Host "[ERROR] Virtual environment not found at .\env." -ForegroundColor Red
    Write-Host "Please run the installer first:" -ForegroundColor Yellow
    Write-Host "    .\install.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate environment in current PowerShell session
& $venvActivate

$env:PYTHONPATH = (Join-Path $PSScriptRoot "src")

if ($Api) {
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host "  Starting FastAPI Medical Coding Service on http://127.0.0.1:8000" -ForegroundColor Green
    Write-Host "===============================================================================" -ForegroundColor Cyan
    & $venvPython -m uvicorn medical_coding.api.app:app --host 0.0.0.0 --port 8000 --reload
} else {
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host "  Starting Streamlit Medical Coding Workspace on http://localhost:8501" -ForegroundColor Green
    Write-Host "===============================================================================" -ForegroundColor Cyan
    & (Join-Path $venvPath "Scripts\streamlit.exe") run (Join-Path $PSScriptRoot "app.py")
}
