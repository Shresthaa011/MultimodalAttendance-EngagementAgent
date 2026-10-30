"""
Pedagogical Recommendations Engine & Teacher Profile Service Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Real-time rule-based alerts (individual distracted student alert, global class warning).
2. Context-aware recommendations (extended disengagement activities, peak highlights, noise cues).
3. Non-prescriptive session teaching suggestions based on temporal profiles.
4. TeacherProfileService matching educational schema (effectiveness score, trend, student trends).
5. SQLAlchemy database initialization and session models.
6. FastAPI REST endpoints (/recommendations/*, /teacher/profile/*) and WebSocket telemetry.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Ensure backend root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.recommendations_engine import RecommendationsEngine
from core.teacher_profiles import TeacherProfileService
from core.database import SessionLocal, Session as SessionModel, engine, Base
from app.main import (
    app,
    recommendations_engine,
    teacher_profile_service,
    get_realtime_recommendations,
    get_teaching_suggestions,
    get_teacher_profile,
    websocket_audio_endpoint,
    RealtimeRecsPayload,
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

def test_realtime_recommendations():
    print("Test 1: Testing real-time recommendations (alerts, warnings, info cues)...")
    engine = RecommendationsEngine()

    # Scenario A: Individual low attention alert & global attention warning
    status_a = {
        "people": [
            {"id": "STU_1", "attention": 35},  # < 40 -> distracted alert
            {"id": "STU_2", "attention": 50},
            {"id": "STU_3", "attention": 55},  # avg = 46.6% < 60% with len > 2 -> global warning
        ],
        "noise_level_db": -45.0,
        "speech_ratio": 0.05,
    }
    recs_a = engine.generate_realtime_recommendations(status_a)
    assert any(r["type"] == "alert" and "STU_1" in r["message"] for r in recs_a), "Failed to detect individual low attention alert"
    assert any(r["type"] == "warning" and "Global class attention is low" in r["message"] for r in recs_a), "Failed to trigger global warning"
    print(f"  [Verified] Alerts Triggered: {[r['message'] for r in recs_a]}")

    # Scenario B: Extended disengagement phase & high noise management
    status_b = {
        "people": [{"id": "STU_1", "attention": 75}, {"id": "STU_2", "attention": 80}],
        "extended_disengagement_phases": [{"start": 0, "end": 4, "duration_sec": 8.0}],
        "noise_level_db": -22.0,
        "noise_stability": "erratic_disturbance",
        "speech_ratio": 0.04,
    }
    recs_b = engine.generate_realtime_recommendations(status_b)
    assert any(r["type"] == "warning" and "Extended low-engagement period detected" in r["message"] for r in recs_b)
    assert any(r["type"] == "warning" and "High environmental noise level" in r["message"] for r in recs_b)
    print(f"  [Verified] Disengagement & Noise Cues: {[r['message'] for r in recs_b]}")

    # Scenario C: Peak engagement recognition
    status_c = {
        "people": [{"id": "STU_1", "attention": 90}],
        "peak_engagement_periods": [{"start": 1, "end": 5, "duration_sec": 10.0}],
        "engagement_score": 88.0,
    }
    recs_c = engine.generate_realtime_recommendations(status_c)
    assert any(r["type"] == "info" and "Effective teaching segment detected" in r["message"] for r in recs_c)
    print(f"  [Verified] Peak Recognition Cue: {[r['message'] for r in recs_c]}")
    print("  [PASSED] Real-time recommendations engine verified.")

def test_session_teaching_suggestions():
    print("\nTest 2: Testing session-level teaching suggestions & positive highlights...")
    engine = RecommendationsEngine()

    mock_profile = [
        {"interval_index": 0, "engagement_score": 0.85},
        {"interval_index": 1, "engagement_score": 0.88},
        {"interval_index": 2, "engagement_score": 0.35},
        {"interval_index": 3, "engagement_score": 0.30},
    ]
    mock_analytics = {
        "mean_engagement_score": 0.62,
        "attention_fluctuation_index": 0.28,
        "engagement_trend": "declining",
        "peak_engagement_periods": [{"start_interval": 0, "end_interval": 1, "duration_sec": 4.0}],
        "extended_disengagement_phases": [{"start_interval": 2, "end_interval": 3, "duration_sec": 4.0}],
    }

    result = engine.generate_session_teaching_suggestions(mock_profile, mock_analytics)
    assert "positive_highlights" in result and len(result["positive_highlights"]) > 0
    assert "teaching_suggestions" in result and len(result["teaching_suggestions"]) > 0
    assert result["session_summary"]["trend"] == "declining"
    print(f"  Session Summary: {result['session_summary']}")
    print(f"  Positive Highlights: {result['positive_highlights']}")
    print(f"  Teaching Suggestions: {[s['suggestion'] for s in result['teaching_suggestions']]}")
    print("  [PASSED] Session-level teaching suggestions successfully generated.")

def test_teacher_profile_service():
    print("\nTest 3: Testing TeacherProfileService schema & metric aggregation...")
    service = TeacherProfileService()
    profile = service.get_profile("TEACHER_404")

    assert profile["teacher_id"] == "TEACHER_404"
    assert profile["effectiveness_score"] == 87.5
    assert profile["trend"] == 12.0
    assert len(profile["class_history"]) >= 3
    assert len(profile["student_trends"]) >= 3
    assert len(profile["strengths"]) >= 2

    # Verify specific items from screenshot
    student_names = [s["name"] for s in profile["student_trends"]]
    assert "Priya" in student_names
    assert "Kumar" in student_names
    assert "Asha" in student_names

    print(f"  Teacher ID: {profile['teacher_id']} | Effectiveness Score: {profile['effectiveness_score']}% | Trend: +{profile['trend']}%")
    print(f"  Class History: {profile['class_history']}")
    print(f"  Student Trends: {profile['student_trends']}")
    print(f"  Strengths: {profile['strengths']}")
    print("  [PASSED] TeacherProfileService output verified.")

def test_database_initialization():
    print("\nTest 4: Testing SQLite database connection & SessionModel ORM...")
    db = SessionLocal()
    try:
        # Check table creation
        tables = list(Base.metadata.tables.keys())
        assert "sessions" in tables
        assert "student_records" in tables
        print(f"  SQLAlchemy Tables Initialized: {tables}")
    finally:
        db.close()
    print("  [PASSED] Database schema and ORM models verified.")

def test_fastapi_recommendations_and_profile_routes():
    print("\nTest 5: Testing FastAPI /recommendations/* & /teacher/profile/* endpoints...")
    # 1. Realtime recommendations endpoint
    payload = RealtimeRecsPayload(session_status={
        "people": [{"id": "STU_10", "attention": 30}],
        "noise_level_db": -22.0,
        "noise_stability": "erratic_disturbance",
        "speech_ratio": 0.02,
    })
    recs_res = drive(get_realtime_recommendations(payload))
    assert recs_res["status"] == "success"
    assert recs_res["total_recommendations"] >= 1
    print(f"  POST /recommendations/realtime -> Recommendations: {len(recs_res['recommendations'])}")

    # 2. Teaching suggestions endpoint
    sugg_res = drive(get_teaching_suggestions())
    assert "teaching_suggestions" in sugg_res
    print(f"  POST /recommendations/teaching-suggestions -> Suggestions Count: {len(sugg_res['teaching_suggestions'])}")

    # 3. Teacher profile endpoint
    prof_res = drive(get_teacher_profile("TEACHER_101"))
    assert prof_res["teacher_id"] == "TEACHER_101"
    assert prof_res["effectiveness_score"] == 87.5
    print(f"  GET /teacher/profile/TEACHER_101 -> Score: {prof_res['effectiveness_score']}% | Trend: +{prof_res['trend']}%")

    # 4. WebSocket streaming containing recommendations
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    pcm = (0.35 * np.sin(2 * np.pi * 320 * t) * 32767.0).astype(np.int16).tobytes()
    ws = MockWebSocket([{"bytes": pcm}])
    drive(websocket_audio_endpoint(ws))
    assert len(ws.sent) >= 1
    ws_msg = ws.sent[0]
    assert "recommendations" in ws_msg
    print(f"  WS /ws/audio Broadcast -> Recommendations Emitted: {len(ws_msg['recommendations'])}")
    print("  [PASSED] FastAPI recommendation endpoints and WebSocket broadcast verified.")

def main():
    print("=" * 70)
    print(" RECOMMENDATIONS ENGINE & TEACHER PROFILE TEST SUITE")
    print("=" * 70)
    try:
        test_realtime_recommendations()
        test_session_teaching_suggestions()
        test_teacher_profile_service()
        test_database_initialization()
        test_fastapi_recommendations_and_profile_routes()
        print("\n" + "=" * 70)
        print("ALL RECOMMENDATIONS & TEACHER PROFILE TESTS PASSED (5/5).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

