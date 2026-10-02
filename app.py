"""Main entrypoint launcher for the Streamlit Medical Coding Application."""

import sys
from pathlib import Path

# Ensure src/ is on Python search path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import and execute main Streamlit UI
from medical_coding.ui.app import *  # noqa: E402, F401, F403
