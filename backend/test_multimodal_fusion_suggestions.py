"""
Multimodal Signal Synchronization, Unified Fusion & AI Suggestions Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Timestamp-based synchronization between asynchronous vision and audio streams.
2. Visual movement pattern tracking (velocity, jitter) and inactivity period accumulation.
3. Proportional multimodal correlation (movement-acoustic states & unified schema).
4. System extensibility (registration of custom modality handlers).
5. AISuggestionEngine (Groq llama3-8b-8192 & pedagogical fallback engine).
6. FastAPI REST endpoints (/fusion/*) and WebSocket unified telemetry streaming.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Ensure backend root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.multimodal_fusion import MultimodalSynchronizer
from core.ai_suggestions import AISuggestionEngine
from app.main import (
    app,
    multimodal_synchronizer,
    ai_suggestion_engine,
    sync_multimodal_signals,
    generate_ai_suggestions,
    get_fusion_state,
    export_fusion_logs,
    websocket_audio_endpoint,
    FusionSyncPayload,
    SuggestionPayload,
)

class MockWebSocket:
    def __init__(self, incoming):
        self.incoming = incoming
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self.incoming:
            return self.incoming.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, data):
        self.sent.append(data)

def drive(coro):
    """Executes a coroutine that only awaits immediate coroutines synchronously without OS loop overhead."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

def test_timestamp_synchronization_and_movement():
    print("Test 1: Testing timestamp-based signal synchronization & movement analysis...")
    syncer = MultimodalSynchronizer(sync_window_sec=1.0)
    t0 = 1000.0

    # Frame 1: Participant 1 at (100, 100)
    v1 = {
        "tracked_participants": [{"track_id": 1, "centroid": (100.0, 100.0), "bbox": (80, 80, 40, 40)}],
        "attendance_records": {"1": {"cumulative_presence_seconds": 1.0, "attendance_status": "ABSENT"}},
        "visual_score": 0.80,
    }
    syncer.register_vision_snapshot(v1, timestamp=t0)

    # Frame 2: 0.5s later, Participant 1 moved to (105, 103) -> velocity ~ 11.6 px/s (moderate natural movement)
    v2 = {
        "tracked_participants": [{"track_id": 1, "centroid": (105.0, 103.0), "bbox": (85, 83, 40, 40)}],
        "attendance_records": {"1": {"cumulative_presence_seconds": 1.5, "attendance_status": "ABSENT"}},
        "visual_score": 0.85,
    }
    rec2 = syncer.register_vision_snapshot(v2, timestamp=t0 + 0.5)
    assert rec2["movement_pattern"] == "moderate_natural_movement"
    assert rec2["mean_velocity_px_sec"] > 5.0
    print(f"  Frame 2 -> Velocity: {rec2['mean_velocity_px_sec']} px/s | Pattern: {rec2['movement_pattern']}")

    # Register concurrent audio at t0 + 0.6
    a1 = {
        "noise_metrics": {"current_noise_level_db": -24.5, "noise_stability": "moderate_fluctuation"},
        "speech_dynamics": {"recent_speech_ratio": 0.35, "interaction_intensity": "high_interaction", "speech_bursts_count": 2},
        "multimodal_fusion_cues": {"classroom_activity_state": "active_discussion", "acoustic_engagement_score": 0.88},
    }
    syncer.register_audio_snapshot(a1, timestamp=t0 + 0.6)

    # Synchronize at t0 + 0.6
    fused = syncer.synchronize_and_fuse(sync_timestamp=t0 + 0.6)
    assert fused["modalities"]["vision"]["movement_pattern"] == "moderate_natural_movement"
    assert fused["modalities"]["audio"]["speech_ratio"] == 0.35
    assert fused["fused_correlation"]["movement_acoustic_state"] == "dynamic_active_presentation"
    print(f"  Fused Snapshot -> State: {fused['fused_correlation']['movement_acoustic_state']} | Proportional MEI: {fused['fused_correlation']['proportional_engagement_score']}")
    print("  [PASSED] Timestamp-based synchronization and movement velocity tracking verified.")

def test_inactivity_and_correlation_states():
    print("\nTest 2: Testing inactivity accumulation & correlation scenarios...")
    syncer = MultimodalSynchronizer(sync_window_sec=1.0)
    t_start = 2000.0

    # Simulate 10 seconds of stationary inactivity (velocity = 0.0)
    for i in range(11):
        v = {
            "tracked_participants": [{"track_id": 2, "centroid": (200.0, 200.0), "bbox": (180, 180, 40, 40)}],
            "attendance_records": {"2": {"cumulative_presence_seconds": float(i), "attendance_status": "PRESENT"}},
            "visual_score": 0.40,
        }
        syncer.register_vision_snapshot(v, timestamp=t_start + float(i))

    # Low audio activity at t_start + 10.0
    a = {
        "noise_metrics": {"current_noise_level_db": -50.0, "noise_stability": "steady"},
        "speech_dynamics": {"recent_speech_ratio": 0.02, "interaction_intensity": "silent_listening"},
        "multimodal_fusion_cues": {"acoustic_engagement_score": 0.35},
    }
    syncer.register_audio_snapshot(a, timestamp=t_start + 10.0)

    fused = syncer.synchronize_and_fuse(sync_timestamp=t_start + 10.0)
    assert fused["modalities"]["vision"]["inactivity_periods_sec"] >= 9.0
    assert fused["fused_correlation"]["movement_acoustic_state"] == "prolonged_passive_inactivity"
    assert fused["fused_correlation"]["correlation_delta"] < 0
    print(f"  Inactivity Period: {fused['modalities']['vision']['inactivity_periods_sec']}s | State: {fused['fused_correlation']['movement_acoustic_state']} | Delta: {fused['fused_correlation']['correlation_delta']}")

    # Now test restless movement + loud disturbance
    t_disturb = t_start + 11.0
    syncer.register_vision_snapshot({
        "tracked_participants": [{"track_id": 2, "centroid": (350.0, 400.0), "bbox": (330, 380, 40, 40)}],
        "visual_score": 0.50,
    }, timestamp=t_disturb)
    syncer.register_audio_snapshot({
        "noise_metrics": {"current_noise_level_db": -18.0, "noise_stability": "erratic_disturbance"},
        "speech_dynamics": {"recent_speech_ratio": 0.04, "interaction_intensity": "silent_listening"},
    }, timestamp=t_disturb)

    fused_disturb = syncer.synchronize_and_fuse(sync_timestamp=t_disturb)
    assert fused_disturb["fused_correlation"]["movement_acoustic_state"] == "restless_environmental_disruption"
    print(f"  Disruption State: {fused_disturb['fused_correlation']['movement_acoustic_state']} | Noise: {fused_disturb['modalities']['audio']['noise_level_db']} dB")
    print("  [PASSED] Inactivity accumulation and multimodal behavioral correlation verified.")

def test_extensibility_architecture():
    print("\nTest 3: Testing extensibility hooks for future modalities...")
    syncer = MultimodalSynchronizer()

    # Register custom biometric wristband handler
    def mock_biometric_handler(timestamp: float) -> dict:
        return {"heart_rate_bpm": 74, "eda_microsiemens": 2.4, "status": "normal_arousal"}

    syncer.register_modality_extension("biometric_wristband", mock_biometric_handler)

    fused = syncer.synchronize_and_fuse(custom_inputs={"clicker_poll_response": "Option B"})
    ext = fused["modalities"]["extensions"]
    assert "biometric_wristband" in ext
    assert ext["biometric_wristband"]["heart_rate_bpm"] == 74
    assert ext["clicker_poll_response"] == "Option B"
    print(f"  Registered Extensions: {list(ext.keys())}")
    print(f"  Biometric Telemetry: {ext['biometric_wristband']}")
    print("  [PASSED] Modality extensibility verified.")

def test_ai_suggestion_engine():
    print("\nTest 4: Testing AISuggestionEngine (llama3-8b-8192 & fallback)...")
    engine = AISuggestionEngine()

    mock_session = {
        "vision": {
            "presence_duration_sec": 120.0,
            "active_participants": 6,
            "movement_pattern": "moderate_natural_movement",
            "mean_velocity_px_sec": 8.5,
            "inactivity_periods_sec": 0.0,
        },
        "audio": {
            "noise_level_db": -24.0,
            "speech_ratio": 0.38,
            "activity_pattern": "active_discussion",
        },
        "fused_correlation": {
            "movement_acoustic_state": "dynamic_active_presentation",
            "proportional_engagement_score": 0.88,
        },
    }

    res = engine.generate_suggestions(mock_session)
    assert res["status"] == "success"
    assert "primary_action" in res and len(res["primary_action"]) > 0
    assert "class_dynamic" in res and len(res["class_dynamic"]) > 0
    assert "suggested_activity" in res and len(res["suggested_activity"]) > 0
    print(f"  Source: {res['source']} (Model: {res.get('model')})")
    print(f"  Primary Action: \"{res['primary_action']}\"")
    print(f"  Class Dynamic: \"{res['class_dynamic']}\"")
    print(f"  Suggested Activity: \"{res['suggested_activity']}\"")
    print("  [PASSED] AISuggestionEngine output and fallback synthesis verified.")

def test_fastapi_fusion_and_websocket_endpoints():
    print("\nTest 5: Testing FastAPI /fusion/* routes & WebSocket streaming...")
    # 1. Sync endpoint
    payload = FusionSyncPayload(
        vision_data={"visual_score": 0.85, "tracked_participants": [{"track_id": 1, "centroid": (100, 100), "bbox": (80, 80, 40, 40)}]},
        audio_data={"noise_metrics": {"current_noise_level_db": -26.0}, "speech_dynamics": {"recent_speech_ratio": 0.32}},
        timestamp=time.time(),
        custom_extensions={"chat_messages_count": 14},
    )
    sync_res = drive(sync_multimodal_signals(payload))
    assert "modalities" in sync_res
    assert "fused_correlation" in sync_res
    print(f"  POST /fusion/sync -> State: {sync_res['fused_correlation']['movement_acoustic_state']} | MEI: {sync_res['fused_correlation']['proportional_engagement_score']}")

    # 2. Suggest endpoint
    suggest_res = drive(generate_ai_suggestions(SuggestionPayload(session_data=sync_res)))
    assert suggest_res["status"] == "success"
    print(f"  POST /fusion/suggest -> Action: \"{suggest_res['primary_action']}\"")

    # 3. State endpoint
    state_res = drive(get_fusion_state())
    assert state_res["total_fused_records"] >= 1
    print(f"  GET /fusion/state -> Total Fused Records: {state_res['total_fused_records']}")

    # 4. Export endpoint
    export_res = drive(export_fusion_logs())
    assert export_res["status"] == "success"
    assert "json_log" in export_res["log_files"]
    print(f"  POST /fusion/export -> JSON Log: {export_res['log_files']['json_log']}")

    # 5. Live WebSocket streaming containing unified_fusion
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    pcm = (0.35 * np.sin(2 * np.pi * 320 * t) * 32767.0).astype(np.int16).tobytes()
    ws = MockWebSocket([{"bytes": pcm}])
    drive(websocket_audio_endpoint(ws))
    assert len(ws.sent) >= 1
    ws_msg = ws.sent[0]
    assert "unified_fusion" in ws_msg
    uf = ws_msg["unified_fusion"]
    print(f"  WS /ws/audio Broadcast -> Unified Fusion State: {uf['fused_correlation']['movement_acoustic_state']}")
    print("  [PASSED] FastAPI Fusion routes and WebSocket unified telemetry streaming verified.")

def main():
    print("=" * 70)
    print(" MULTIMODAL AGENT - SIGNAL SYNCHRONIZATION & SUGGESTIONS TEST SUITE")
    print("=" * 70)
    try:
        test_timestamp_synchronization_and_movement()
        test_inactivity_and_correlation_states()
        test_extensibility_architecture()
        test_ai_suggestion_engine()
        test_fastapi_fusion_and_websocket_endpoints()
        print("\n" + "=" * 70)
        print("ALL MULTIMODAL SYNCHRONIZATION & SUGGESTION TESTS PASSED (5/5).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
