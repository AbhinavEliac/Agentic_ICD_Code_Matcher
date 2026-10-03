@echo off
setlocal

set "VENV_DIR=env"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .\%VENV_DIR%.
    echo Please run the installer first:
    echo     install.bat
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
    )
)

set PYTHONPATH=%CD%\src;%PYTHONPATH%

if "%1"=="--api" (
    echo ===============================================================================
    echo   Starting FastAPI Medical Coding Service on http://127.0.0.1:8000
    echo ===============================================================================
    python -m uvicorn medical_coding.api.app:app --host 0.0.0.0 --port 8000 --reload
) else (
    echo ===============================================================================
    echo   Starting Streamlit Medical Coding Workspace on http://localhost:8501
    echo ===============================================================================
    streamlit run app.py
)
