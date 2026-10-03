#!/usr/bin/env bash
# Linux / macOS Run Script
set -e

VENV_DIR="env"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found at ./$VENV_DIR."
    echo "Please run the installer first:"
    echo "    ./install.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
    fi
fi

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

if [ "$1" = "--api" ]; then
    echo "==============================================================================="
    echo "  Starting FastAPI Medical Coding Service on http://127.0.0.1:8000"
    echo "==============================================================================="
    python -m uvicorn medical_coding.api.app:app --host 0.0.0.0 --port 8000 --reload
else
    echo "==============================================================================="
    echo "  Starting Streamlit Medical Coding Workspace on http://localhost:8501"
    echo "==============================================================================="
    streamlit run app.py
fi
