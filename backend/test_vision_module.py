"""
Vision Module Verification & Benchmark Test
Multimodal Attendance & Engagement Agent
Tests Face Detection, Spatial Tracking, and Sustained Presence Calculation.
"""

import sys
import time
import numpy as np
import cv2

from core.face_detection import FaceDetector
from core.tracker import FaceTracker
from core.presence_tracker import PresenceTracker
from core.vision_service import VisionService

def generate_synthetic_face_frame(width=640, height=480, face_positions=[(150, 150, 90, 90)]):
    """
    Generates a synthetic frame with realistic gradient features resembling faces
    to allow deterministic testing of the pipeline without requiring physical cameras.
    """
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    for (x, y, w, h) in face_positions:
        # Draw head oval
        cv2.ellipse(frame, (x + w // 2, y + h // 2), (w // 2, int(h * 0.65)), 0, 0, 360, (200, 200, 200), -1)
        # Draw eyes
        eye_y = y + int(h * 0.4)
        cv2.circle(frame, (x + int(w * 0.3), eye_y), int(w * 0.08), (20, 20, 20), -1)
        cv2.circle(frame, (x + int(w * 0.7), eye_y), int(w * 0.08), (20, 20, 20), -1)
        # Draw mouth
        mouth_y = y + int(h * 0.75)
        cv2.ellipse(frame, (x + w // 2, mouth_y), (int(w * 0.25), int(h * 0.1)), 0, 0, 180, (20, 20, 20), -1)
    return frame

def test_face_detector_initialization():
    print("Test 1: Initializing FaceDetector (Haarcascade)...")
    detector = FaceDetector()
    assert detector.face_cascade is not None, "CascadeClassifier is None"
    assert not detector.face_cascade.empty(), "CascadeClassifier failed to load haarcascade XML"
    print("  [PASSED] Haarcascade classifier initialized successfully from OpenCV data path.")

def test_tracker_spatial_association():
    print("\nTest 2: Testing FaceTracker spatial association across moving frames...")
    tracker = FaceTracker(max_disappeared=5, iou_threshold=0.2, distance_threshold=60.0)

    # Frame 1: Participant at (100, 100)
    det1 = [{"bbox": (100, 100, 60, 60), "score": 1.0}]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    p1_id = tracks1[0]["participant_id"]
    print(f"  Frame 1: Registered {p1_id} at {tracks1[0]['bbox']}")

    # Frame 2: Participant shifts slightly to (105, 103) - typical head movement
    det2 = [{"bbox": (105, 103, 60, 60), "score": 1.0}]
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    assert tracks2[0]["participant_id"] == p1_id, "Track ID changed despite continuous movement!"
    print(f"  Frame 2: Associated {tracks2[0]['participant_id']} at {tracks2[0]['bbox']} (ID Preserved: True)")

    # Frame 3: A second participant appears at (350, 200)
    det3 = [
        {"bbox": (110, 105, 60, 60), "score": 1.0},
        {"bbox": (350, 200, 60, 60), "score": 1.0},
    ]
    tracks3 = tracker.update(det3)
    assert len(tracks3) == 2
    ids = {t["participant_id"] for t in tracks3}
    assert p1_id in ids, f"Original participant {p1_id} lost upon multi-person entry!"
    print(f"  Frame 3: Multi-person tracking verified. Active IDs: {ids}")

    # Frame 4: Brief occlusion (Participant 1 disappears for 2 frames)
    det4 = [{"bbox": (352, 201, 60, 60), "score": 1.0}]
    tracks4 = tracker.update(det4)
    # Participant 1 is in occlusion memory (disappeared_frames = 1 <= max_disappeared)
    assert len(tracks4) == 1
    assert tracks4[0]["participant_id"] != p1_id

    # Frame 5: Participant 1 reappears at (115, 108)
    det5 = [
        {"bbox": (115, 108, 60, 60), "score": 1.0},
        {"bbox": (355, 202, 60, 60), "score": 1.0},
    ]
    tracks5 = tracker.update(det5)
    reappeared_ids = {t["participant_id"] for t in tracks5}
    assert p1_id in reappeared_ids, "Participant failed to re-associate after brief occlusion!"
    print(f"  Frame 4-5: Occlusion recovery verified. Track {p1_id} maintained across disappearance.")
    print("  [PASSED] FaceTracker identity consistency and occlusion recovery passed.")

def test_presence_tracker_and_attendance():
    print("\nTest 3: Testing PresenceTracker duration accumulation & attendance status...")
    presence_tracker = PresenceTracker(min_attendance_threshold_sec=0.15)  # low threshold for testing

    tracked_p1 = [{"participant_id": "participant_1", "track_id": 1, "bbox": (100, 100, 50, 50)}]

    # Simulate 5 consecutive frames over 200ms
    for i in range(5):
        time.sleep(0.04)
        summary = presence_tracker.update(tracked_p1)

    assert len(summary) == 1
    record = summary[0]
    print(f"  Accumulated Presence: {record['presence_duration_seconds']}s (Frames: {record['total_frames_detected']})")
    print(f"  Attendance Status: {record['attendance_status']}")
    assert record["presence_duration_seconds"] > 0.1, "Presence duration failed to accumulate!"
    assert record["attendance_status"] == "PRESENT", f"Expected PRESENT, got {record['attendance_status']}"
    print("  [PASSED] Presence duration calculation & attendance criteria evaluation passed.")

def test_end_to_end_vision_service():
    print("\nTest 4: Testing VisionService end-to-end integration...")
    service = VisionService(camera_id=0, min_attendance_threshold_sec=0.1)

    # Feed synthetic frames
    frame1 = generate_synthetic_face_frame(face_positions=[(120, 120, 80, 80)])
    res1 = service.run_on_frame(frame1)

    assert "frame_id" in res1
    assert "presence_summary" in res1
    print(f"  Processed frame {res1['frame_id']} | Participants detected: {res1['participants_count']}")
    print("  [PASSED] VisionService end-to-end pipeline execution passed.")

def main():
    print("=" * 70)
    print(" MULTIMODAL AGENT - VISION MODULE VALIDATION & INTEGRATION TEST")
    print("=" * 70)
    try:
        test_face_detector_initialization()
        test_tracker_spatial_association()
        test_presence_tracker_and_attendance()
        test_end_to_end_vision_service()
        print("\n" + "=" * 70)
        print("ALL VISION MODULE TESTS PASSED (4/4).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
