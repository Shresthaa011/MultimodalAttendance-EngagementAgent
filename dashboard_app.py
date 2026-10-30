"""
Streamlit Dashboard Root Shortcut
Multimodal Attendance & Engagement Agent
"""
import runpy
from pathlib import Path

target = Path(__file__).parent / "dashboard" / "app.py"
runpy.run_path(str(target), run_name="__main__")

