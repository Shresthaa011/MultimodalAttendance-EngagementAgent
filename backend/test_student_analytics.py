"""
Student Analytics & Reflective Session Trends Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Session trends aggregation (minute/second timeline buckets, average attention, dominant emotion).
2. Student-facing reflective analytics (presence duration, consistency index, focus/distraction timelines).
3. Non-evaluative feedback generation promoting self-awareness without grading or peer ranking.
4. Privacy preservation and lack of peer comparison metrics.
5. FastAPI REST endpoints (/analytics/*) and database integration.
"""

import sys
import time
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Ensure backend root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.analytics_service import AnalyticsService
from core.database import SessionLocal, SessionPerson, PersonMetric
from app.main import (
    app,
    analytics_service,
    get_session_trends_endpoint,
    get_student_reflective_endpoint,
    record_metric_endpoint,
    RecordMetricPayload,
)

def drive(coro):
    """Executes a coroutine synchronously."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

def test_session_trends_aggregation():
    print("Test 1: Testing session trends aggregation across time buckets...")
    service = AnalyticsService()
    session_id = 999

    now = datetime.utcnow()
    # Insert sample metrics
    service.record_person_metric(session_id, "STU_A", 85.0, "engaged", timestamp=now)
    service.record_person_metric(session_id, "STU_B", 75.0, "neutral", timestamp=now)
    service.record_person_metric(session_id, "STU_A", 60.0, "neutral", timestamp=now + timedelta(seconds=2))
    service.record_person_metric(session_id, "STU_B", 90.0, "engaged", timestamp=now + timedelta(seconds=2))

    trends = service.get_session_trends(session_id)
    assert trends["session_id"] == session_id
    assert trends["total_data_points"] >= 4
    assert len(trends["timeline"]) >= 2

    first_bucket = trends["timeline"][0]
    assert "timestamp" in first_bucket
    assert "average_attention" in first_bucket
    assert first_bucket["average_attention"] == 80.0  # (85 + 75) / 2
    assert first_bucket["samples_count"] == 2
    print(f"  Timeline Bucket 1: Time={first_bucket['timestamp']} | Avg Attention={first_bucket['average_attention']}% | Emotion={first_bucket['dominant_emotion']}")
    print("  [PASSED] Session trends aggregation verified.")

def test_student_reflective_summary():
    print("\nTest 2: Testing student reflective summary & self-awareness analytics...")
    service = AnalyticsService()
    session_id = 888
    student_id = "STU_PRIYA"

    # Enroll student in session
    service.record_session_person(
        session_id=session_id,
        person_id=student_id,
        name="Priya Sharma",
        presence_duration=1200.0,  # 20 minutes
        attendance_status="PRESENT",
        average_attention=82.0,
    )

    # Ingest temporal telemetry: 3 high focus points, 2 drop points, 2 recovery points
    base_time = datetime(2026, 10, 30, 10, 0, 0)
    scores = [85.0, 90.0, 88.0, 35.0, 30.0, 78.0, 82.0]
    for idx, att in enumerate(scores):
        service.record_person_metric(
            session_id=session_id,
            person_id=student_id,
            attention=att,
            emotion="engaged" if att >= 75 else "confused",
            timestamp=base_time + timedelta(seconds=idx * 15),
        )

    summary = service.get_student_reflective_summary(session_id, student_id)

    # 1. Attendance checks
    assert summary["student_id"] == student_id
    assert summary["student_name"] == "Priya Sharma"
    assert summary["attendance_summary"]["attendance_status"] == "PRESENT"
    assert summary["attendance_summary"]["presence_duration_minutes"] == 20.0

    # 2. Consistency checks
    assert "average_attention_score" in summary["engagement_consistency"]
    assert "attention_stability" in summary["engagement_consistency"]
    print(f"  Engagement Consistency: Score={summary['engagement_consistency']['average_attention_score']}% | Stability={summary['engagement_consistency']['attention_stability']}")

    # 3. Focus and distraction timelines
    assert len(summary["activity_timeline"]["high_focus_periods"]) >= 1
    assert len(summary["activity_timeline"]["attention_drop_periods"]) >= 1
    print(f"  High Focus Periods: {summary['activity_timeline']['high_focus_periods']}")
    print(f"  Attention Drop Periods: {summary['activity_timeline']['attention_drop_periods']}")

    # 4. Reflective feedback (non-grading)
    assert len(summary["reflective_feedback"]) >= 1
    for fb in summary["reflective_feedback"]:
        print(f"  Reflective Feedback: \"{fb}\"")

    # 5. Privacy guarantees
    assert "privacy_notice" in summary
    assert "rank" not in summary
    assert "grade" not in summary
    print(f"  Privacy Notice: \"{summary['privacy_notice']}\"")
    print("  [PASSED] Student reflective analytics successfully generated.")

def test_sparse_student_fallback():
    print("\nTest 3: Testing sparse student data fallback...")
    service = AnalyticsService()
    summary = service.get_student_reflective_summary(session_id=777, student_id="STU_NEW")
    assert summary["student_id"] == "STU_NEW"
    assert "attendance_summary" in summary
    assert "reflective_feedback" in summary
    print(f"  Fallback Summary Generated for: {summary['student_name']}")
    print("  [PASSED] Sparse student fallback verified.")

def test_fastapi_analytics_endpoints():
    print("\nTest 4: Testing FastAPI /analytics/* REST routes...")
    # 1. Record metric endpoint
    rec_res = drive(record_metric_endpoint(RecordMetricPayload(
        session_id=601,
        person_id="STU_KUMAR",
        attention=88.5,
        emotion="engaged",
    )))
    assert rec_res["status"] == "success"
    print("  POST /analytics/record-metric -> Success")

    # 2. Session trends endpoint
    trends_res = drive(get_session_trends_endpoint(601))
    assert trends_res["session_id"] == 601
    assert trends_res["total_data_points"] >= 1
    print(f"  GET /analytics/session-trends/601 -> Points: {trends_res['total_data_points']}")

    # 3. Student reflective summary endpoint
    stud_res = drive(get_student_reflective_endpoint(601, "STU_KUMAR"))
    assert stud_res["student_id"] == "STU_KUMAR"
    assert "attendance_summary" in stud_res
    assert "reflective_feedback" in stud_res
    print(f"  GET /analytics/student-reflective/601/STU_KUMAR -> Score: {stud_res['engagement_consistency']['average_attention_score']}%")
    print("  [PASSED] FastAPI student analytics routes verified.")

def main():
    print("=" * 70)
    print(" STUDENT ANALYTICS & REFLECTIVE SESSION TRENDS TEST SUITE")
    print("=" * 70)
    try:
        test_session_trends_aggregation()
        test_student_reflective_summary()
        test_sparse_student_fallback()
        test_fastapi_analytics_endpoints()
        print("\n" + "=" * 70)
        print("ALL STUDENT ANALYTICS TESTS PASSED (4/4).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

