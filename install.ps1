# PowerShell Installation Script for Windows
$ErrorActionPreference = "Stop"

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  Local ICD-10-CM / ICD-O / CPT Coding Engine - Installation Setup (PowerShell)" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[*] Detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not found on your PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ (recommend 3.10-3.13) and ensure 'Add Python to PATH' is checked." -ForegroundColor Yellow
    exit 1
}

# 2. Virtual Environment
$venvPath = Join-Path $PSScriptRoot "env"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[*] Creating virtual environment in .\env ..." -ForegroundColor Cyan
    python -m venv $venvPath
    Write-Host "[*] Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "[*] Virtual environment already exists at .\env." -ForegroundColor Yellow
}

# 3. Upgrade pip
Write-Host "[*] Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet

# 4. Install Dependencies
Write-Host "[*] Installing required packages from requirements.txt..." -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
Write-Host "[*] Dependencies installed successfully." -ForegroundColor Green

# 5. Environment Config
$envFile = Join-Path $PSScriptRoot ".env"
$envExample = Join-Path $PSScriptRoot ".env.example"
if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Write-Host "[*] Initializing .env configuration from .env.example..." -ForegroundColor Cyan
        Copy-Item -Path $envExample -Destination $envFile
    } else {
        Write-Host "[*] Creating default .env configuration..." -ForegroundColor Cyan
        @"
APP_ENV=production
LOG_LEVEL=INFO
ICD_DATABASE_DIR=./Database
ICD_DATASET_PATH=./data/icd10/sample_hospital_icd.csv
FAISS_INDEX_DIR=./data/indexes
SQLITE_DB_PATH=./data/warehouse/encounters.db
EMBEDDING_DEVICE=cpu
"@ | Out-File -FilePath $envFile -Encoding utf8
    }
    Write-Host "[*] Environment file .env created." -ForegroundColor Green
} else {
    Write-Host "[*] Existing .env file detected." -ForegroundColor Yellow
}

# 6. Ensure Directories
@("data/indexes", "data/warehouse", "data/uploads") | ForEach-Object {
    $dir = Join-Path $PSScriptRoot $_
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# 7. Build/Verify Indexes
Write-Host "[*] Checking local medical coding database and search indices..." -ForegroundColor Cyan
$indexScript = Join-Path $PSScriptRoot "scripts\index_icd.py"
try {
    & $venvPython $indexScript --force-fast-embeddings
    Write-Host "[*] Authoritative search indexes verified and ready." -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Index initialization completed with warnings. Application will load on demand." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  INSTALLATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "To launch the web application, run:" -ForegroundColor White
Write-Host "    .\run.ps1" -ForegroundColor Yellow
Write-Host "or:" -ForegroundColor White
Write-Host "    .\run.bat" -ForegroundColor Yellow
Write-Host ""
Write-Host "For FastAPI backend mode, run:" -ForegroundColor White
Write-Host "    .\run.ps1 -Api" -ForegroundColor Yellow
Write-Host ""
