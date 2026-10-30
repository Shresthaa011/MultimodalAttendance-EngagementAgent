"""
Attendance Record Management & Presence-Based Logging
Multimodal Attendance & Engagement Agent

Translates continuous face tracking data into meaningful attendance records.
Uses cumulative presence-based logic to prevent false positives from brief appearances,
accumulating intermittent sightings and persisting structured JSON and CSV reports.
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

class AttendanceRecord:
    """Represents a validated attendance record for an individual participant."""
    def __init__(self, participant_id: str, track_id: int, initial_timestamp: float):
        self.participant_id = participant_id
        self.track_id = track_id
        self.student_name: Optional[str] = None
        self.first_seen = initial_timestamp
        self.last_seen = initial_timestamp
        self.cumulative_presence_seconds: float = 0.0
        self.total_frames_detected: int = 1
        self.attendance_status: str = "ABSENT"
        self._last_tick_timestamp: float = initial_timestamp
        self.presence_segments: List[Dict[str, float]] = [{"start": initial_timestamp, "end": initial_timestamp}]

    def update_presence(self, current_timestamp: float, min_threshold_sec: float, segment_gap_threshold_sec: float = 1.0):
        """Accumulates intermittent presence duration and updates attendance status."""
        delta = current_timestamp - self._last_tick_timestamp

        # Intermittent presence accumulation:
        # If consecutive frames arrive within segment_gap_threshold_sec, add exact elapsed delta.
        # Otherwise, treat as new intermittent appearance segment after absence.
        if 0 < delta <= segment_gap_threshold_sec:
            self.cumulative_presence_seconds += delta
            self.presence_segments[-1]["end"] = current_timestamp
        else:
            self.cumulative_presence_seconds += 0.033  # baseline single-frame detection
            self.presence_segments.append({"start": current_timestamp, "end": current_timestamp})

        self.last_seen = current_timestamp
        self._last_tick_timestamp = current_timestamp
        self.total_frames_detected += 1

        # Confirm attendance once cumulative presence threshold is met
        if self.cumulative_presence_seconds >= min_threshold_sec:
            self.attendance_status = "PRESENT"

    def associate_student_identity(self, student_id: str, student_name: Optional[str] = None):
        """Maps anonymous track ID to verified student profile."""
        self.participant_id = student_id
        self.student_name = student_name or student_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "track_id": self.track_id,
            "student_name": self.student_name,
            "first_seen_iso": datetime.fromtimestamp(self.first_seen).strftime("%Y-%m-%d %H:%M:%S"),
            "last_seen_iso": datetime.fromtimestamp(self.last_seen).strftime("%Y-%m-%d %H:%M:%S"),
            "first_seen_timestamp": round(self.first_seen, 3),
            "last_seen_timestamp": round(self.last_seen, 3),
            "cumulative_presence_seconds": round(self.cumulative_presence_seconds, 2),
            "presence_duration_seconds": round(self.cumulative_presence_seconds, 2),
            "total_frames_detected": self.total_frames_detected,
            "attendance_status": self.attendance_status,
            "appearance_segments_count": len(self.presence_segments),
        }

class AttendanceManager:
    """
    Manages session-level attendance logic based on sustained visual presence.
    Generates structured attendance logs and persistent analytical exports.
    """
    def __init__(
        self,
        session_id: Optional[str] = None,
        min_presence_threshold_sec: float = 10.0,
        segment_gap_threshold_sec: float = 1.0,
        log_dir: str = "logs/attendance"
    ):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.min_presence_threshold_sec = min_presence_threshold_sec
        self.segment_gap_threshold_sec = segment_gap_threshold_sec
        self.log_dir = log_dir
        self.session_start_timestamp = time.time()
        self.records: Dict[str, AttendanceRecord] = {}

        # Ensure storage directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def process_frame_tracking(self, tracked_participants: List[Dict], current_timestamp: Optional[float] = None) -> List[Dict]:
        """
        Translates raw tracker output from current frame into accumulated attendance records.
        """
        current_time = current_timestamp if current_timestamp is not None else time.time()

        for item in tracked_participants:
            pid = item.get("participant_id", f"participant_{item['track_id']}")
            tid = item["track_id"]

            if pid not in self.records:
                self.records[pid] = AttendanceRecord(pid, tid, current_time)
            else:
                self.records[pid].update_presence(current_time, self.min_presence_threshold_sec, self.segment_gap_threshold_sec)

        return self.get_attendance_summary()

    def identify_participant(self, track_id: int, student_id: str, student_name: Optional[str] = None):
        """Associates a tracked participant with a recognized student profile."""
        old_pid = f"participant_{track_id}"
        if old_pid in self.records:
            record = self.records.pop(old_pid)
            record.associate_student_identity(student_id, student_name)
            self.records[student_id] = record

    def get_attendance_summary(self) -> List[Dict[str, Any]]:
        """Returns structured attendance entries for all observed participants."""
        return [record.to_dict() for record in self.records.values()]

    def save_structured_logs(self) -> Dict[str, str]:
        """
        Persists structured attendance records into JSON and CSV log files.
        Returns paths to created log files.
        """
        summary_list = self.get_attendance_summary()
        present_count = sum(1 for r in summary_list if r["attendance_status"] == "PRESENT")
        absent_count = len(summary_list) - present_count

        log_payload = {
            "session_id": self.session_id,
            "session_start_iso": datetime.fromtimestamp(self.session_start_timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "exported_at_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "min_presence_threshold_sec": self.min_presence_threshold_sec,
            "total_participants": len(summary_list),
            "present_count": present_count,
            "absent_count": absent_count,
            "attendance_rate_percent": round((present_count / max(1, len(summary_list))) * 100.0, 1),
            "attendance_records": summary_list,
        }

        # 1. JSON Structured Log
        json_filename = f"attendance_log_{self.session_id}.json"
        json_path = os.path.join(self.log_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)

        # 2. CSV Tabular Report
        csv_filename = f"attendance_report_{self.session_id}.csv"
        csv_path = os.path.join(self.log_dir, csv_filename)
        if summary_list:
            df = pd.DataFrame(summary_list)
            df.to_csv(csv_path, index=False)
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("participant_id,track_id,cumulative_presence_seconds,attendance_status\n")

        return {
            "json_log_path": json_path,
            "csv_report_path": csv_path,
        }
