"""
Unified Vision Service
Multimodal Attendance & Engagement Agent
Orchestrates Camera Ingestion, Face Detection, Face Recognition,
Persistent Tracking, and Presence-Based Attendance Logging.
"""

import cv2
import time
from typing import Dict, List, Optional, Tuple
from .face_detection import FaceDetector
from .tracker import FaceTracker
from .presence_tracker import PresenceTracker
from .camera_service import CameraService
from .face_recognition import FaceRecognizer
from .attendance_manager import AttendanceManager

class VisionService:
    """
    High-level orchestrator for the Vision Module.
    Extracts frames, detects faces, identifies students, tracks identities,
    and maintains structured attendance records.
    """
    def __init__(
        self,
        camera_id: int = 0,
        min_attendance_threshold_sec: float = 10.0,
        student_db_path: str = "student_db",
        log_dir: str = "logs/attendance",
        recognition_interval_frames: int = 15,
    ):
        self.camera_service = CameraService(camera_id=camera_id)
        self.detector = FaceDetector()
        self.tracker = FaceTracker(max_disappeared=15, iou_threshold=0.3, distance_threshold=80.0)
        self.presence_tracker = PresenceTracker(min_attendance_threshold_sec=min_attendance_threshold_sec)
        self.recognizer = FaceRecognizer(db_path=student_db_path)
        self.attendance_manager = AttendanceManager(
            min_presence_threshold_sec=min_attendance_threshold_sec,
            log_dir=log_dir,
        )
        self.recognition_interval_frames = recognition_interval_frames
        self.frame_count = 0
        self.start_time = time.time()
        self.track_to_student: Dict[int, str] = {}

    def start(self):
        self.camera_service.start()

    def stop(self):
        self.camera_service.stop()

    def process_single_frame(self, frame) -> Tuple[any, List[Dict], List[Dict]]:
        """
        Processes a single video frame:
        1. Detects faces via Haarcascade
        2. Tracks identities across frames
        3. Periodically identifies enrolled students via DeepFace
        4. Accumulates intermittent presence in AttendanceManager
        5. Annotates the frame with bounding boxes and attendance badges
        Returns: (annotated_frame, tracked_participants, attendance_summary)
        """
        self.frame_count += 1

        # Step 1: Face Detection
        detections = self.detector.process_frame(frame)

        # Step 2: Spatial & IoU-based Tracking
        tracked = self.tracker.update(detections)

        # Step 3: Face Recognition (run periodically to conserve computation)
        if self.frame_count % self.recognition_interval_frames == 1 and len(tracked) > 0:
            recognized_faces = self.recognizer.recognize(frame)
            for rf in recognized_faces:
                # Match recognized face bbox to tracked participant
                rx, ry, rw, rh = rf["bbox"]
                r_center = (rx + rw / 2.0, ry + rh / 2.0)
                for p in tracked:
                    px, py, pw, ph = p["bbox"]
                    if px <= r_center[0] <= px + pw and py <= r_center[1] <= py + ph:
                        tid = p["track_id"]
                        sid = rf["student_id"]
                        self.track_to_student[tid] = sid
                        p["participant_id"] = sid
                        self.attendance_manager.identify_participant(tid, sid)

        # Apply known student IDs to tracked participants
        for p in tracked:
            tid = p["track_id"]
            if tid in self.track_to_student:
                p["participant_id"] = self.track_to_student[tid]

        # Step 4: Sustained Presence & Attendance Accumulation
        presence_summary = self.attendance_manager.process_frame_tracking(tracked)

        # Also update legacy presence tracker for backward compatibility
        self.presence_tracker.update(tracked)

        # Step 5: Visual Annotation
        annotated_frame = self.annotate_frame(frame.copy(), tracked)

        return annotated_frame, tracked, presence_summary

    def annotate_frame(self, frame, tracked_participants: List[Dict]):
        """Draws participant IDs, bounding boxes, and presence status badges onto the frame."""
        for p in tracked_participants:
            x, y, w, h = p["bbox"]
            pid = p["participant_id"]
            track_id = p["track_id"]

            # Query attendance status for badge color
            record = self.attendance_manager.records.get(pid)
            is_present = record and record.attendance_status == "PRESENT"
            color = (0, 255, 127) if is_present else (0, 165, 255)  # Green if PRESENT, Orange if in-progress

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Header label badge
            presence_sec = round(record.cumulative_presence_seconds, 1) if record else 0.0
            status_text = "PRESENT" if is_present else f"{presence_sec}s"
            label = f"{pid} [{status_text}]"

            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Centroid point
            cx, cy = int(p["centroid"][0]), int(p["centroid"][1])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Global Session HUD overlay
        active_present = sum(1 for r in self.attendance_manager.records.values() if r.attendance_status == "PRESENT")
        fps_text = f"Frames: {self.frame_count} | Participants: {len(tracked_participants)} | Present: {active_present}"
        cv2.putText(frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def run_on_frame(self, frame) -> Dict:
        """Lightweight API handler returning structured detection & attendance data."""
        annotated_frame, tracked, presence = self.process_single_frame(frame)
        return {
            "frame_id": self.frame_count,
            "timestamp": time.time(),
            "participants_count": len(tracked),
            "tracked_participants": tracked,
            "attendance_records": presence,
            "presence_summary": presence,
        }

    def export_attendance(self) -> Dict[str, str]:
        """Saves structured JSON and CSV attendance logs to disk."""
        return self.attendance_manager.save_structured_logs()
