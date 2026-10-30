"""
Attendance Logic & Presence Accumulation Verification Test
Multimodal Attendance & Engagement Agent

Validates:
1. Intermittent appearance accumulation across time.
2. Prevention of false positives from brief appearances.
3. Transition from ABSENT to PRESENT once cumulative presence threshold is exceeded.
4. Structured logging in JSON and CSV formats.
5. FaceRecognizer registration and student identification mapping.
"""

import os
import sys
import time
import json
import numpy as np
import cv2

from core.attendance_manager import AttendanceManager
from core.face_recognition import FaceRecognizer
from core.vision_service import VisionService

def test_intermittent_presence_accumulation():
    print("Test 1: Testing intermittent presence accumulation & threshold crossing...")
    mgr = AttendanceManager(
        session_id="test_session_intermittent",
        min_presence_threshold_sec=0.25,
        segment_gap_threshold_sec=0.08,
        log_dir="logs/test_attendance"
    )

    # Segment 1: Participant 1 appears for 3 ticks (~0.12s)
    t0 = time.time()
    for _ in range(3):
        mgr.process_frame_tracking([{"track_id": 1, "participant_id": "participant_1"}])
        time.sleep(0.04)

    record = mgr.records["participant_1"]
    print(f"  Segment 1: Presence={record.cumulative_presence_seconds:.2f}s | Status={record.attendance_status}")
    assert record.attendance_status == "ABSENT", "Participant prematurely marked present on brief appearance!"

    # Simulate absence gap of 2 seconds
    time.sleep(0.1)

    # Segment 2: Participant 1 returns for 5 ticks (~0.20s), total should exceed 0.25s
    for _ in range(5):
        mgr.process_frame_tracking([{"track_id": 1, "participant_id": "participant_1"}])
        time.sleep(0.04)

    record = mgr.records["participant_1"]
    print(f"  Segment 2: Cumulative Presence={record.cumulative_presence_seconds:.2f}s | Status={record.attendance_status}")
    assert record.cumulative_presence_seconds >= 0.25, "Cumulative presence failed to accumulate across intermittent segments!"
    assert record.attendance_status == "PRESENT", f"Expected PRESENT, got {record.attendance_status}"
    assert len(record.presence_segments) >= 2, "Intermittent segments were not recorded!"
    print("  [PASSED] Intermittent appearance accumulation and threshold transition verified.")

def test_false_positive_prevention():
    print("\nTest 2: Testing false-positive prevention for accidental/brief appearances...")
    mgr = AttendanceManager(session_id="test_session_fp", min_presence_threshold_sec=5.0, log_dir="logs/test_attendance")

    # Accidental appearance: passerby detected for only 2 frames (approx 0.07s)
    mgr.process_frame_tracking([{"track_id": 99, "participant_id": "passerby_99"}])
    time.sleep(0.033)
    mgr.process_frame_tracking([{"track_id": 99, "participant_id": "passerby_99"}])

    record = mgr.records["passerby_99"]
    print(f"  Passerby Presence: {record.cumulative_presence_seconds:.3f}s | Status: {record.attendance_status}")
    assert record.attendance_status == "ABSENT", "False positive occurred: brief detection marked as PRESENT!"
    print("  [PASSED] False-positive prevention verified: brief appearances remain ABSENT.")

def test_structured_log_export():
    print("\nTest 3: Testing structured JSON and CSV attendance logging...")
    mgr = AttendanceManager(session_id="test_session_export", min_presence_threshold_sec=0.1, log_dir="logs/test_attendance")

    # Add 2 participants: one PRESENT, one ABSENT
    for _ in range(4):
        mgr.process_frame_tracking([{"track_id": 101, "participant_id": "student_alice"}])
        time.sleep(0.04)
    # Bob is only seen once
    mgr.process_frame_tracking([{"track_id": 102, "participant_id": "student_bob"}])

    log_files = mgr.save_structured_logs()
    json_path = log_files["json_log_path"]
    csv_path = log_files["csv_report_path"]

    assert os.path.exists(json_path), f"JSON log not found at {json_path}"
    assert os.path.exists(csv_path), f"CSV report not found at {csv_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["session_id"] == "test_session_export"
    assert data["total_participants"] == 2
    assert data["present_count"] == 1
    assert data["absent_count"] == 1
    print(f"  JSON Log Created: {json_path} (Present: {data['present_count']}, Absent: {data['absent_count']})")
    print(f"  CSV Report Created: {csv_path}")
    print("  [PASSED] Structured JSON and CSV attendance logs successfully generated.")

def test_face_recognizer_and_identity_mapping():
    print("\nTest 4: Testing FaceRecognizer student registration & identity mapping...")
    db_path = "student_db"
    recognizer = FaceRecognizer(db_path=db_path)
    assert os.path.exists(db_path), "Student DB directory failed to initialize"

    # Create dummy reference face image
    dummy_face = np.full((100, 100, 3), 180, dtype=np.uint8)
    registered = recognizer.register_student(dummy_face, "STU_1001")
    assert registered, "Failed to register student image"
    assert os.path.exists(os.path.join(db_path, "STU_1001.jpg")), "Reference image not written to student_db"
    print("  Registered student 'STU_1001' into student_db successfully.")

    # Test identity mapping in AttendanceManager
    mgr = AttendanceManager(session_id="test_session_id_map", min_presence_threshold_sec=0.1, log_dir="logs/test_attendance")
    mgr.process_frame_tracking([{"track_id": 5, "participant_id": "participant_5"}])
    assert "participant_5" in mgr.records

    # Map track 5 to recognized STU_1001
    mgr.identify_participant(track_id=5, student_id="STU_1001", student_name="Alice Smith")
    assert "STU_1001" in mgr.records
    assert mgr.records["STU_1001"].student_name == "Alice Smith"
    print(f"  Mapped track_id 5 -> {mgr.records['STU_1001'].participant_id} ({mgr.records['STU_1001'].student_name})")
    print("  [PASSED] FaceRecognizer registration and identity association verified.")

def main():
    print("=" * 70)
    print(" MULTIMODAL AGENT - PRESENCE-BASED ATTENDANCE LOGIC VERIFICATION")
    print("=" * 70)
    try:
        test_intermittent_presence_accumulation()
        test_false_positive_prevention()
        test_structured_log_export()
        test_face_recognizer_and_identity_mapping()
        print("\n" + "=" * 70)
        print("ALL PRESENCE-BASED ATTENDANCE TESTS PASSED (4/4).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
