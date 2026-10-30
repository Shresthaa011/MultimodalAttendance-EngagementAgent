"""
Basic Pipeline Smoke Test
Multimodal Attendance & Engagement Agent
Validates imports, basic operations, and framework health.
"""

import sys
import numpy as np

def test_imports():
    print("Testing module imports...")
    import cv2
    import mediapipe as mp
    import deep_sort_realtime
    import librosa
    import soundfile as sf
    import fastapi
    import uvicorn
    import websockets
    import sqlalchemy
    import pandas as pd
    import groq
    import dotenv
    print("All modules imported successfully!")

def test_cv_operation():
    print("Testing basic OpenCV & NumPy operation...")
    import cv2
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Test Stream", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    assert dummy_frame.shape == (480, 640, 3)
    print("OpenCV frame creation & draw operation: PASSED")

def test_audio_operation():
    print("Testing basic audio processing with NumPy & Librosa...")
    import librosa
    sr = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 440 Hz sine wave (musical note A4)
    audio_signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rms = librosa.feature.rms(y=audio_signal)
    assert rms.shape[1] > 0
    print(f"Librosa audio feature extraction (RMS shape={rms.shape}): PASSED")

def test_fastapi_app():
    print("Testing FastAPI app initialization...")
    from app.main import app
    assert app.title == "Multimodal Attendance & Engagement Agent API"
    print("FastAPI application configuration: PASSED")

def main():
    print("=" * 60)
    print(" MULTIMODAL AGENT - INTEGRATION & SMOKE TEST")
    print("=" * 60)
    try:
        test_imports()
        test_cv_operation()
        test_audio_operation()
        test_fastapi_app()
        print("=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY! ENVIRONMENT IS READY.")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
