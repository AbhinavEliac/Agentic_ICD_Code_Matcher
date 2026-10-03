"""Main entrypoint launcher for the Streamlit Medical Coding Application."""

import sys
from pathlib import Path

# Ensure src/ is on Python search path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Automatically detect and include local virtual environment site-packages
for venv_name in ("env", ".venv", "venv"):
    # Windows
    win_sp = root_dir / venv_name / "Lib" / "site-packages"
    if win_sp.exists() and str(win_sp) not in sys.path:
        sys.path.insert(0, str(win_sp))
    # Linux / macOS
    linux_lib = root_dir / venv_name / "lib"
    if linux_lib.exists():
        for py_dir in linux_lib.glob("python*"):
            sp = py_dir / "site-packages"
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))

# Import and execute main Streamlit UI reliably across browser refreshes
import runpy

runpy.run_module("medical_coding.ui.app", run_name="__main__")
