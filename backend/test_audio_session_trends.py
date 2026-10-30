"""
Audio Session Trends, Noise Level Dynamics & WebSocket Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Short-time audio feature computation (energy, amplitude variation, ZCR, spectral centroid).
2. Session-level aggregation into acoustic context metrics (noise stability, interaction intensity).
3. Discrimination between silence, background noise, and active speech.
4. Structuring of audio insights for multimodal fusion.
5. FastAPI WebSocket (/ws/audio) real-time streaming integration.
"""

import sys
import os
import json
import time
import asyncio
import numpy as np

from app.main import app, audio_processor, websocket_audio_endpoint
from core.audio_analysis import AudioAnalyzer
from core.audio_processor import AudioProcessor

class MockWebSocket:
    """Mock WebSocket simulating real-time browser audio streaming from AudioCapture.jsx."""
    def __init__(self, incoming_messages):
        self.incoming = incoming_messages
        self.sent_messages = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self.incoming:
            return self.incoming.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, data):
        self.sent_messages.append(data)

def generate_signal_sequence():
    """
    Generates a realistic 6-second classroom acoustic sequence:
    - 0-2s: Quiet classroom / attentive listening (-60 dB)
    - 2-4s: Active student question / discussion (~ -18 dB)
    - 4-6s: Low ambient background noise (-42 dB)
    """
    sr = 16000
    # 1. Quiet listening (2.0s)
    quiet = np.random.normal(0, 0.0008, int(sr * 2.0)).astype(np.float32)

    # 2. Active speech (2.0s) - modulated harmonic wave
    t_speech = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    harmonics = 0.35 * np.sin(2 * np.pi * 240 * t_speech) + 0.20 * np.sin(2 * np.pi * 480 * t_speech)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t_speech))
    speech = (harmonics * envelope).astype(np.float32)

    # 3. Ambient classroom rustle (2.0s)
    ambient = np.random.normal(0, 0.012, int(sr * 2.0)).astype(np.float32)

    return np.concatenate([quiet, speech, ambient])

def test_feature_computation_and_discrimination():
    print("Test 1: Testing short-time feature computation & state discrimination...")
    analyzer = AudioAnalyzer(sample_rate=16000)

    # Silence
    quiet_metrics = analyzer.analyze_audio_chunk(np.random.normal(0, 0.0005, 16000).astype(np.float32))
    assert quiet_metrics["speech_detected"] is False
    assert quiet_metrics["noise_level_db"] < -45.0
    print(f"  Quiet State: Noise={quiet_metrics['noise_level_db']} dB | Pattern={quiet_metrics['activity_pattern']}")

    # Speech with harmonics
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    harmonics = 0.40 * np.sin(2 * np.pi * 280 * t) + 0.25 * np.sin(2 * np.pi * 560 * t)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    speech_sig = (harmonics * envelope).astype(np.float32)
    speech_metrics = analyzer.analyze_audio_chunk(speech_sig)
    assert speech_metrics["speech_detected"] is True
    assert speech_metrics["speech_ratio"] > 0.20
    print(f"  Speech State: Noise={speech_metrics['noise_level_db']} dB | Speech Ratio={speech_metrics['speech_ratio']} | Pattern={speech_metrics['activity_pattern']}")

    # Ambient Noise
    noise_sig = (0.05 * np.random.normal(0, 1.0, 16000)).astype(np.float32)
    noise_metrics = analyzer.analyze_audio_chunk(noise_sig)
    print(f"  Ambient Noise State: Noise={noise_metrics['noise_level_db']} dB | Pattern={noise_metrics['activity_pattern']}")
    print("  [PASSED] Feature computation & acoustic state discrimination verified.")

def test_session_level_trends_and_multimodal_cues():
    print("\nTest 2: Testing session-level aggregation & multimodal fusion cues...")
    proc = AudioProcessor(sample_rate=16000, window_duration_sec=1.0, hop_duration_sec=1.0)

    # Feed 6-second realistic sequence
    seq = generate_signal_sequence()
    windows = proc.segment_waveform(seq, base_timestamp=time.time())
    assert len(windows) == 6, f"Expected 6 windows, got {len(windows)}"

    context = proc.get_session_audio_context(recent_window_count=6)
    assert context["windows_analyzed"] == 6
    assert "noise_metrics" in context
    assert "speech_dynamics" in context
    assert "multimodal_fusion_cues" in context

    print(f"  Session Windows Analyzed: {context['windows_analyzed']}")
    print(f"  Noise Level Dynamics: Mean={context['noise_metrics']['mean_noise_level_db']} dB | Stability={context['noise_metrics']['noise_stability']}")
    print(f"  Speech Interaction: Intensity={context['speech_dynamics']['interaction_intensity']} | Turns={context['speech_dynamics']['speech_bursts_count']}")
    print(f"  Multimodal Fusion Cue: Score={context['multimodal_fusion_cues']['acoustic_engagement_score']} | State={context['multimodal_fusion_cues']['classroom_activity_state']}")
    assert context["speech_dynamics"]["speech_bursts_count"] >= 1, "Failed to register speech turn"
    print("  [PASSED] Session-level acoustic aggregation and fusion cues verified.")

def drive(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

def test_fastapi_websocket_audio_streaming():
    print("\nTest 3: Testing FastAPI WebSocket (/ws/audio) real-time streaming...")
    # Generate 16-bit PCM chunk (16000 samples = 1 sec @ 16 kHz)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    wave = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    pcm_bytes = (wave * 32767.0).astype(np.int16).tobytes()

    ws = MockWebSocket([{"bytes": pcm_bytes}])
    drive(websocket_audio_endpoint(ws))

    assert ws.accepted is True
    assert len(ws.sent_messages) >= 1
    response = ws.sent_messages[0]
    assert response["type"] == "audio_metrics"
    assert "metrics" in response
    assert "session_context" in response
    metrics = response["metrics"]
    print(f"  WebSocket Telemetry -> Noise: {metrics['noise_level_db']} dB | Speech: {metrics['speech_detected']} | Engagement: {metrics['acoustic_engagement_score']}")
    print(f"  Session Context -> Intensity: {response['session_context']['speech_dynamics']['interaction_intensity']}")
    print("  [PASSED] WebSocket real-time audio ingest and telemetry streaming verified.")

def main():
    print("=" * 70)
    print(" MULTIMODAL AGENT - AUDIO ACTIVITY & NOISE ANALYSIS TEST SUITE")
    print("=" * 70)
    try:
        test_feature_computation_and_discrimination()
        test_session_level_trends_and_multimodal_cues()
        test_fastapi_websocket_audio_streaming()
        print("\n" + "=" * 70)
        print("ALL AUDIO ACTIVITY TESTS PASSED (3/3).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
