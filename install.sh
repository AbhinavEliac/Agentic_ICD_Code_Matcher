#!/usr/bin/env bash
# Linux / macOS Installation Script
set -e

echo "==============================================================================="
echo "  Local ICD-10-CM / ICD-O / CPT Coding Engine - Installation Setup (Linux)"
echo "==============================================================================="
echo ""

# 1. Check Python
PYTHON_BIN=""
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "[ERROR] Python 3 was not found on your PATH."
    echo "Please install Python 3.10+ (e.g., sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

PY_VER=$($PYTHON_BIN --version 2>&1)
echo "[*] Detected: $PY_VER"

# 2. Virtual Environment
VENV_DIR="env"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[*] Creating virtual environment in ./$VENV_DIR ..."
    $PYTHON_BIN -m venv "$VENV_DIR"
    echo "[*] Virtual environment created successfully."
else
    echo "[*] Virtual environment already exists at ./$VENV_DIR."
fi

# 3. Upgrade pip
echo "[*] Upgrading pip, setuptools, and wheel..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel --quiet

# 4. Install Dependencies
echo "[*] Installing required packages from requirements.txt..."
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
echo "[*] Dependencies installed successfully."

# 5. Environment Config
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[*] Initializing .env configuration from .env.example..."
        cp .env.example .env
    else
        echo "[*] Creating default .env configuration..."
        cat <<EOF > .env
APP_ENV=production
LOG_LEVEL=INFO
ICD_DATABASE_DIR=./Database
ICD_DATASET_PATH=./data/icd10/sample_hospital_icd.csv
FAISS_INDEX_DIR=./data/indexes
SQLITE_DB_PATH=./data/warehouse/encounters.db
EMBEDDING_DEVICE=cpu
EOF
    fi
    echo "[*] Environment file .env created."
else
    echo "[*] Existing .env file detected."
fi

# 6. Ensure Directories
mkdir -p data/indexes data/warehouse data/uploads

# 7. Build/Verify Indexes
echo "[*] Checking local medical coding database and search indices..."
"$VENV_DIR/bin/python" scripts/index_icd.py --force-fast-embeddings || {
    echo "[WARNING] Index initialization completed with warnings. Application will load on demand."
}

echo ""
echo "==============================================================================="
echo "  INSTALLATION COMPLETED SUCCESSFULLY!"
echo "==============================================================================="
echo "To launch the web application, run:"
echo "    ./run.sh"
echo ""
echo "For FastAPI backend mode, run:"
echo "    ./run.sh --api"
echo ""
