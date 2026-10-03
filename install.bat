@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   Local ICD-10-CM / ICD-O / CPT Coding Engine - Installation Setup (Windows)
echo ===============================================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found on your PATH.
    echo Please install Python 3.10+ (recommend 3.10-3.13) and ensure "Add Python to PATH" is checked.
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [*] Detected: %PYTHON_VERSION%

:: 2. Create Virtual Environment
set "VENV_DIR=env"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [*] Creating virtual environment in .\%VENV_DIR% ...
    python -m venv %VENV_DIR%
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [*] Virtual environment created successfully.
) else (
    echo [*] Virtual environment already exists at .\%VENV_DIR%.
)

:: 3. Upgrade pip and core packaging tools
echo [*] Upgrading pip, setuptools, and wheel...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel >nul 2>nul

:: 4. Install Dependencies
echo [*] Installing required packages from requirements.txt...
echo     (This may take a few minutes depending on network speed)
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    exit /b 1
)
echo [*] Dependencies installed successfully.

:: 5. Set up Environment Variables (.env)
if not exist ".env" (
    if exist ".env.example" (
        echo [*] Initializing .env configuration from .env.example...
        copy /y ".env.example" ".env" >nul
    ) else (
        echo [*] Creating default .env configuration...
        (
            echo APP_ENV=production
            echo LOG_LEVEL=INFO
            echo ICD_DATABASE_DIR=./Database
            echo ICD_DATASET_PATH=./data/icd10/sample_hospital_icd.csv
            echo FAISS_INDEX_DIR=./data/indexes
            echo SQLITE_DB_PATH=./data/warehouse/encounters.db
            echo EMBEDDING_DEVICE=cpu
        ) > ".env"
    )
    echo [*] Environment file .env created.
) else (
    echo [*] Existing .env file detected. Keeping existing configuration.
)

:: 6. Ensure required directories exist
if not exist "data\indexes" mkdir data\indexes
if not exist "data\warehouse" mkdir data\warehouse
if not exist "data\uploads" mkdir data\uploads

:: 7. Build/Verify Local Search Indexes
echo [*] Checking local medical coding database and search indices...
"%VENV_DIR%\Scripts\python.exe" scripts\index_icd.py --force-fast-embeddings
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Index initialization completed with warnings. Application will load on demand.
) else (
    echo [*] Authoritative search indexes verified and ready.
)

echo.
echo ===============================================================================
echo   INSTALLATION COMPLETED SUCCESSFULLY!
echo ===============================================================================
echo.
echo The application and environment are fully configured.
echo To launch the web application, run:
echo     run.bat
echo or:
echo     run.ps1
echo.
echo For FastAPI backend mode, run:
echo     run.bat --api
echo.
exit /b 0
