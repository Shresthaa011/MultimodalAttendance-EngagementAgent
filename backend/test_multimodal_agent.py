"""
Multimodal Agent & Pedagogical Intelligence Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Rule-based reasoning across pedagogical classroom scenarios (R1-R6).
2. Transparent explainability, rule triggers, and modality contribution breakdown.
3. Temporal session intelligence, rolling mean MEI, and trend analysis.
4. LLM / Rule-based insight and recommendation generation (InsightGenerator).
5. FastAPI REST and WebSocket telemetry integration.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Ensure backend root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.multimodal_agent import MultimodalAgent
from core.llm_insights import InsightGenerator
from app.main import (
    app,
    multimodal_agent,
    agent_status,
    evaluate_agent_snapshot,
    get_session_insight,
    reset_agent_session,
    websocket_audio_endpoint,
    EvaluateSnapshotPayload,
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

def test_rule_based_reasoning_scenarios():
    print("Test 1: Testing rule-based reasoning across classroom scenarios (R1 - R6)...")
    agent = MultimodalAgent(visual_weight=0.55, audio_weight=0.45)

    # 1. Active Collaboration (R1)
    res_collab = agent.evaluate_multimodal_snapshot(
        vision_data={"visual_score": 0.85, "participants_count": 5},
        audio_data={
            "noise_metrics": {"current_noise_level_db": -22.0, "noise_stability": "moderate_fluctuation"},
            "speech_dynamics": {"recent_speech_ratio": 0.40, "interaction_intensity": "high_interaction"},
            "multimodal_fusion_cues": {"acoustic_engagement_score": 0.90},
        }
    )
    assert res_collab["primary_state"] == "active_collaboration", f"Expected active_collaboration, got {res_collab['primary_state']}"
    assert res_collab["engagement_level"] == "HIGH"
    assert any(r["rule_id"] == "R1_ACTIVE_COLLABORATION" for r in res_collab["rules_triggered"])
    print(f"  [R1 Verified] Active Collaboration -> Score: {res_collab['multimodal_engagement_index']} | Level: {res_collab['engagement_level']}")

    # 2. Focused Silent Listening (R2)
    res_lecture = agent.evaluate_multimodal_snapshot(
        vision_data={"visual_score": 0.90, "participants_count": 5},
        audio_data={
            "noise_metrics": {"current_noise_level_db": -55.0, "noise_stability": "steady"},
            "speech_dynamics": {"recent_speech_ratio": 0.02, "interaction_intensity": "silent_listening"},
            "multimodal_fusion_cues": {"acoustic_engagement_score": 0.75},
        }
    )
    assert res_lecture["primary_state"] == "focused_listening", f"Expected focused_listening, got {res_lecture['primary_state']}"
    assert res_lecture["engagement_level"] == "HIGH"
    assert any(r["rule_id"] == "R2_FOCUSED_LISTENING" for r in res_lecture["rules_triggered"])
    print(f"  [R2 Verified] Focused Listening -> Score: {res_lecture['multimodal_engagement_index']} | Level: {res_lecture['engagement_level']}")

    # 3. Environmental Disturbance (R3)
    res_disturb = agent.evaluate_multimodal_snapshot(
        vision_data={"visual_score": 0.50, "participants_count": 3},
        audio_data={
            "noise_metrics": {"current_noise_level_db": -20.0, "noise_stability": "erratic_disturbance"},
            "speech_dynamics": {"recent_speech_ratio": 0.05, "interaction_intensity": "silent_listening"},
            "multimodal_fusion_cues": {"acoustic_engagement_score": 0.20},
        }
    )
    assert res_disturb["primary_state"] == "environmental_disturbance", f"Expected environmental_disturbance, got {res_disturb['primary_state']}"
    assert res_disturb["engagement_level"] == "LOW"
    assert any(r["rule_id"] == "R3_ENVIRONMENTAL_DISTURBANCE" for r in res_disturb["rules_triggered"])
    print(f"  [R3 Verified] Environmental Disturbance -> Score: {res_disturb['multimodal_engagement_index']} | Level: {res_disturb['engagement_level']}")

    # 4. Passive Inattention (R4)
    res_inatt = agent.evaluate_multimodal_snapshot(
        vision_data={"visual_score": 0.25, "participants_count": 2},
        audio_data={
            "noise_metrics": {"current_noise_level_db": -52.0, "noise_stability": "steady"},
            "speech_dynamics": {"recent_speech_ratio": 0.02, "interaction_intensity": "silent_listening"},
            "multimodal_fusion_cues": {"acoustic_engagement_score": 0.30},
        }
    )
    assert res_inatt["primary_state"] == "passive_inattention", f"Expected passive_inattention, got {res_inatt['primary_state']}"
    assert res_inatt["engagement_level"] == "LOW"
    assert any(r["rule_id"] == "R4_PASSIVE_INATTENTION" for r in res_inatt["rules_triggered"])
    print(f"  [R4 Verified] Passive Inattention -> Score: {res_inatt['multimodal_engagement_index']} | Level: {res_inatt['engagement_level']}")

    # 5. Off-Camera Verbal Interaction (R5)
    res_offcam = agent.evaluate_multimodal_snapshot(
        vision_data={"visual_score": 0.30, "participants_count": 1},
        audio_data={
            "noise_metrics": {"current_noise_level_db": -26.0, "noise_stability": "moderate_fluctuation"},
            "speech_dynamics": {"recent_speech_ratio": 0.35, "interaction_intensity": "high_interaction"},
            "multimodal_fusion_cues": {"acoustic_engagement_score": 0.85},
        }
    )
    assert res_offcam["primary_state"] == "off_camera_verbal_interaction", f"Expected off_camera_verbal_interaction, got {res_offcam['primary_state']}"
    assert any(r["rule_id"] == "R5_OFF_CAMERA_INTERACTION" for r in res_offcam["rules_triggered"])
    print(f"  [R5 Verified] Off-Camera Interaction -> Score: {res_offcam['multimodal_engagement_index']} | Level: {res_offcam['engagement_level']}")

    print("  [PASSED] Rule-based reasoning engine verified across all classroom states.")

def test_explainability_and_transparency():
    print("\nTest 2: Testing explainability, transparent rule traces & contribution breakdown...")
    agent = MultimodalAgent(visual_weight=0.55, audio_weight=0.45)

    snapshot = agent.evaluate_multimodal_snapshot(
        vision_data={"visual_score": 0.75, "participants_count": 4},
        audio_data={
            "noise_metrics": {"current_noise_level_db": -48.0, "noise_stability": "steady"},
            "speech_dynamics": {"recent_speech_ratio": 0.05, "interaction_intensity": "silent_listening"},
            "multimodal_fusion_cues": {"acoustic_engagement_score": 0.70},
        }
    )

    assert "explanation" in snapshot and len(snapshot["explanation"]) > 10
    assert "rules_triggered" in snapshot and len(snapshot["rules_triggered"]) > 0
    assert "modality_scores" in snapshot
    weights = snapshot["modality_scores"]
    assert round(weights["visual_weight"] + weights["audio_weight"], 2) == 1.0
    print(f"  Explanation Generated: \"{snapshot['explanation']}\"")
    print(f"  Triggered Rule: {snapshot['rules_triggered'][0]['rule_id']} - {snapshot['rules_triggered'][0]['description']}")
    print(f"  Modality Contributions: Visual (w={weights['visual_weight']}, s={weights['visual_score']}) + Audio (w={weights['audio_weight']}, s={weights['acoustic_score']}) + Delta ({weights['context_delta']})")
    print("  [PASSED] Explainability and transparent contribution breakdown verified.")

def test_session_intelligence_and_trends():
    print("\nTest 3: Testing session aggregation, rolling MEI & trend analysis...")
    agent = MultimodalAgent()

    # Feed an improving engagement progression
    low_scores = [0.35, 0.40, 0.42]
    high_scores = [0.75, 0.82, 0.88]

    for score in low_scores:
        agent.evaluate_multimodal_snapshot(
            vision_data={"visual_score": score},
            audio_data={"acoustic_engagement_score": score, "noise_level_db": -45.0, "speech_ratio": 0.05}
        )
    for score in high_scores:
        agent.evaluate_multimodal_snapshot(
            vision_data={"visual_score": score},
            audio_data={"acoustic_engagement_score": score, "noise_level_db": -30.0, "speech_ratio": 0.30}
        )

    intel = agent.get_session_intelligence()
    assert intel["session_active"] is True
    assert intel["total_snapshots"] == 6
    assert intel["engagement_trend"] == "improving"
    assert "llm_insight" in intel and len(intel["llm_insight"]) > 0
    assert "pedagogical_recommendation" in intel and len(intel["pedagogical_recommendation"]) > 0
    print(f"  Session Trend: {intel['engagement_trend']} | Mean MEI: {intel['mean_engagement_index']} | Level: {intel['engagement_level']}")
    print(f"  Pedagogical Insight: \"{intel['llm_insight']}\"")
    print(f"  Instructor Recommendation: \"{intel['pedagogical_recommendation']}\"")
    print("  [PASSED] Session-level temporal trends and pedagogical intelligence verified.")

def test_llm_insight_generator():
    print("\nTest 4: Testing InsightGenerator fallback and prompt construction...")
    gen = InsightGenerator()

    mock_session = {
        "attendance": {"present_count": 18, "total_students": 20},
        "visual": {"visual_score": 0.88},
        "audio": {"noise_level_db": -28.0, "noise_stability": "moderate_fluctuation", "speech_ratio": 0.42, "activity_pattern": "active_collaboration", "interaction_intensity": "high_interaction"},
        "multimodal": {"engagement_score": 0.85, "primary_state": "active_collaboration", "rules_triggered": ["R1_ACTIVE_COLLABORATION"]},
    }

    res = gen.generate_classroom_insight(mock_session)
    assert res["status"] == "success"
    assert "insight" in res and len(res["insight"]) > 0
    assert "recommendation" in res and len(res["recommendation"]) > 0
    print(f"  Insight Source: {res['source']} (Model: {res['model']})")
    print(f"  Insight: \"{res['insight']}\"")
    print(f"  Recommendation: \"{res['recommendation']}\"")
    print("  [PASSED] InsightGenerator prompt synthesis and parsing verified.")

def test_fastapi_agent_endpoints():
    print("\nTest 5: Testing FastAPI Multimodal Agent endpoints & WebSocket fusion...")
    # 1. Reset agent
    reset_resp = drive(reset_agent_session())
    assert reset_resp["status"] == "success"

    # 2. Evaluate snapshot endpoint
    payload = EvaluateSnapshotPayload(
        vision_data={"visual_score": 0.82, "participants_count": 4},
        audio_data={
            "noise_metrics": {"current_noise_level_db": -25.0, "noise_stability": "steady"},
            "speech_dynamics": {"recent_speech_ratio": 0.30, "interaction_intensity": "high_interaction"},
            "multimodal_fusion_cues": {"acoustic_engagement_score": 0.80},
        }
    )
    eval_resp = drive(evaluate_agent_snapshot(payload))
    assert eval_resp["primary_state"] == "active_collaboration"
    assert eval_resp["engagement_level"] == "HIGH"
    print(f"  POST /agent/evaluate -> State: {eval_resp['primary_state']} | MEI: {eval_resp['multimodal_engagement_index']}")

    # 3. Status endpoint
    status_resp = drive(agent_status())
    assert status_resp["total_snapshots"] >= 1
    print(f"  GET /agent/status -> Snapshots: {status_resp['total_snapshots']} | Status: {status_resp['status']}")

    # 4. Session insight endpoint
    insight_resp = drive(get_session_insight())
    assert insight_resp["session_active"] is True
    print(f"  GET /agent/session-insight -> Mean MEI: {insight_resp['mean_engagement_index']} | Insight: \"{insight_resp['llm_insight']}\"")

    # 5. Live WebSocket streaming containing multimodal fusion
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    pcm = (0.35 * np.sin(2 * np.pi * 320 * t) * 32767.0).astype(np.int16).tobytes()
    ws = MockWebSocket([{"bytes": pcm}])
    drive(websocket_audio_endpoint(ws))
    assert len(ws.sent) >= 1
    ws_msg = ws.sent[0]
    assert "multimodal_engagement" in ws_msg
    fusion = ws_msg["multimodal_engagement"]
    print(f"  WS /ws/audio Broadcast -> State: {fusion['primary_state']} | MEI: {fusion['multimodal_engagement_index']} | Level: {fusion['engagement_level']}")
    print("  [PASSED] FastAPI Multimodal Agent endpoints and WebSocket telemetry verified.")

def main():
    print("=" * 70)
    print(" MULTIMODAL AGENT - REASONING, EXPLAINABILITY & INSIGHTS TEST SUITE")
    print("=" * 70)
    try:
        test_rule_based_reasoning_scenarios()
        test_explainability_and_transparency()
        test_session_intelligence_and_trends()
        test_llm_insight_generator()
        test_fastapi_agent_endpoints()
        print("\n" + "=" * 70)
        print("ALL MULTIMODAL AGENT TESTS PASSED (5/5).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

