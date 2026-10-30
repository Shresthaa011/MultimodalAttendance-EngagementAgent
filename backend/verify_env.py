"""
Environment Verification Script
Multimodal Attendance & Engagement Agent
Milestone 1: Environment & Dependency Validation
"""

import sys
import importlib
from typing import Dict, Tuple

MODULE_CATEGORIES: Dict[str, Dict[str, str]] = {
    "Computer Vision": {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "deep_sort_realtime": "deep-sort-realtime",
        "deepface": "deepface",
        "werkzeug": "werkzeug",
    },
    "Audio Processing": {
        "librosa": "librosa",
        "soundfile": "soundfile",
    },
    "Backend & API": {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "websockets": "websockets",
        "multipart": "python-multipart",
        "sqlalchemy": "sqlalchemy",
        "dotenv": "python-dotenv",
    },
    "Data Processing & LLM": {
        "numpy": "numpy",
        "pandas": "pandas",
        "groq": "groq",
    },
}

def verify_module(module_name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed (no __version__)")
        return True, str(version)
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 70)
    print(" MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT - ENVIRONMENT VERIFIER")
    print("=" * 70)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print("=" * 70)

    total_modules = 0
    passed_modules = 0
    missing = []

    for category, modules in MODULE_CATEGORIES.items():
        print(f"\n[Category: {category}]")
        print("-" * 50)
        for import_name, package_name in modules.items():
            total_modules += 1
            ok, info = verify_module(import_name)
            if ok:
                passed_modules += 1
                status = " OK "
                print(f"  [{status}] {package_name:<20} (import: {import_name:<18}) v{info}")
            else:
                status = "FAIL"
                print(f"  [{status}] {package_name:<20} (import: {import_name:<18}) -> {info}")
                missing.append((package_name, import_name))

    print("\n" + "=" * 70)
    print(f"Summary: {passed_modules}/{total_modules} packages verified.")
    if missing:
        print("\nMissing packages to install:")
        print("  pip install " + " ".join(pkg for pkg, _ in missing))
        print("\nOr install everything via:")
        print("  pip install -r requirements.txt")
        print("=" * 70)
        return 1
    else:
        print("All dependencies are successfully installed and verified!")
        print("=" * 70)
        return 0

if __name__ == "__main__":
    sys.exit(main())
