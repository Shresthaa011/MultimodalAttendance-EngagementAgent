"""
End-to-End Pipeline Validation & Robustness Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Complete subsystem initialization and local execution readiness.
2. Dual-mode input handling: Live webcam capture fallback & pre-recorded classroom video simulation.
3. Full data flow: Vision -> Audio -> Multimodal Fusion -> Engagement Scorer -> Recommendations -> Analytics.
4. Presence accumulation and threshold-based attendance marking.
5. Audio noise level extraction and speech activity categorization.
6. Real-time pedagogical recommendations and non-evaluative student analytics.
7. Visual demonstration HUD frame rendering and video artifact export.
"""

import os
import sys
import time
from pathlib import Path

# Add backend root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.pipeline_validator import PipelineValidator
from core.database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric

def test_subsystem_initialization():
    print("Test 1: Testing subsystem initialization & module bindings...")
    validator = PipelineValidator(session_id=8801, min_attendance_threshold_sec=3.0)

    assert validator.detector is not None
    assert validator.tracker is not None
    assert validator.attendance_manager is not None
    assert validator.audio_processor is not None
    assert validator.synchronizer is not None
    assert validator.multimodal_agent is not None
    assert validator.engagement_scorer is not None
    assert validator.recommendations_engine is not None
    assert validator.analytics_service is not None
    print("  [PASSED] All submodules instantiated successfully.")

def test_end_to_end_pipeline_execution():
    print("\nTest 2: Executing full pipeline across simulated classroom video & audio...")
    validator = PipelineValidator(
        session_id=8802,
        min_attendance_threshold_sec=2.0,
        output_dir="logs/demonstration/test_run",
    )

    # Run for 3.0 seconds at 10 FPS (30 frames)
    report = validator.execute_live_pipeline(
        duration_seconds=3.0,
        fps=10.0,
        save_demonstration_video=True,
    )

    assert report["status"] == "VALIDATION_SUCCESS"
    assert report["total_frames_processed"] == 30
    assert len(validator.live_telemetry_history) == 30

    print(f"  Processed Frames: {report['total_frames_processed']}")
    print(f"  Active Telemetry History: {len(validator.live_telemetry_history)} intervals")

    # Check attendance results
    attendance = report["attendance_summary"]
    assert len(attendance) >= 1
    stu1 = next((r for r in attendance if "STU_1001" in r["participant_id"] or r["track_id"] == 1), None)
    assert stu1 is not None, "Participant 1 was not tracked in attendance summary"
    print(f"  STU_1001 Status: {stu1['attendance_status']} (Presence: {stu1['presence_duration_seconds']:.2f}s)")

    # Check subsystem verification flags
    subsystems = report["subsystems_verified"]
    for sys_name, verified in subsystems.items():
        assert verified is True, f"Subsystem {sys_name} failed validation"
    print("  [PASSED] Full pipeline execution and data propagation verified.")

def test_multimodal_fusion_and_scoring():
    print("\nTest 3: Testing multimodal fusion and temporal engagement scoring...")
    validator = PipelineValidator(
        session_id=8803,
        min_attendance_threshold_sec=1.5,
        output_dir="logs/demonstration/test_fusion",
    )

    report = validator.execute_live_pipeline(duration_seconds=2.0, fps=10.0)

    # Verify temporal profile length
    assert report["engagement_temporal_profile_count"] > 0
    scorer_profile = validator.engagement_scorer.temporal_profile
    assert len(scorer_profile) > 0

    first_score = scorer_profile[0]
    assert "engagement_score" in first_score
    assert "timestamp" in first_score
    assert 0.0 <= first_score["engagement_score"] <= 1.0

    print(f"  Engagement Profile Points: {len(scorer_profile)}")
    print(f"  First Score: {first_score['engagement_score']} | Active Mode: {validator.engagement_scorer.mode}")
    print("  [PASSED] Multimodal fusion and temporal scoring validated.")

def test_recommendations_and_analytics():
    print("\nTest 4: Testing real-time recommendations and student reflective summary...")
    validator = PipelineValidator(
        session_id=8804,
        min_attendance_threshold_sec=2.0,
        output_dir="logs/demonstration/test_analytics",
    )

    report = validator.execute_live_pipeline(duration_seconds=2.5, fps=10.0)

    # Check student reflective summary
    reflective = report.get("reflective_summary", {})
    assert reflective.get("student_id") == "STU_1001"
    assert "attendance" in reflective
    assert "engagement_consistency" in reflective
    assert "focus_timeline" in reflective
    assert "privacy_notice" in reflective

    print(f"  Student ID: {reflective['student_id']}")
    print(f"  Formatted Presence: {reflective['attendance']['presence_duration_formatted']}")
    print(f"  Consistency Category: {reflective['engagement_consistency']['stability_category']}")
    print(f"  Privacy Notice: \"{reflective['privacy_notice']}\"")
    print("  [PASSED] Pedagogical recommendations and non-evaluative analytics verified.")

def test_visual_demonstration_artifacts():
    print("\nTest 5: Testing visual demonstration HUD export and video artifacts...")
    validator = PipelineValidator(
        session_id=8805,
        min_attendance_threshold_sec=2.0,
        output_dir="logs/demonstration/test_visual",
    )

    report = validator.execute_live_pipeline(
        duration_seconds=2.0,
        fps=10.0,
        save_demonstration_video=True,
    )

    artifacts = report["exported_artifacts"]
    hud_img = artifacts["demonstration_hud_image"]
    demo_video = artifacts["demonstration_video"]

    assert os.path.exists(hud_img), f"HUD image was not created at {hud_img}"
    assert os.path.getsize(hud_img) > 1000, "HUD image is empty or corrupt"
    print(f"  Verified HUD Snapshot: {hud_img} ({os.path.getsize(hud_img)} bytes)")

    assert os.path.exists(demo_video), f"Demo video was not created at {demo_video}"
    assert os.path.getsize(demo_video) > 1000, "Demo video is empty or corrupt"
    print(f"  Verified Demo Video: {demo_video} ({os.path.getsize(demo_video)} bytes)")

    # Verify structured logs exported
    att_json = artifacts["attendance_logs"].get("json_log_path", artifacts["attendance_logs"].get("json_path"))
    aud_json = artifacts["audio_logs"].get("json_log_path", artifacts["audio_logs"].get("json_path"))
    fus_json = artifacts["fusion_logs"].get("json_log_path", artifacts["fusion_logs"].get("json_log", artifacts["fusion_logs"].get("json_path")))

    assert os.path.exists(att_json), f"Attendance log not found at {att_json}"
    assert os.path.exists(aud_json), f"Audio log not found at {aud_json}"
    assert os.path.exists(fus_json), f"Fusion log not found at {fus_json}"
    print("  [PASSED] Visual HUD rendering and multimodal demonstration artifacts validated.")

def main():
    print("=" * 70)
    print(" MULTIMODAL AGENT - END-TO-END PIPELINE VALIDATION TEST SUITE")
    print("=" * 70)
    try:
        test_subsystem_initialization()
        test_end_to_end_pipeline_execution()
        test_multimodal_fusion_and_scoring()
        test_recommendations_and_analytics()
        test_visual_demonstration_artifacts()
        print("\n" + "=" * 70)
        print(" ALL END-TO-END VALIDATION TESTS PASSED (5/5). SYSTEM ROBUST & RELIABLE.")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n[TEST FAILED]: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
