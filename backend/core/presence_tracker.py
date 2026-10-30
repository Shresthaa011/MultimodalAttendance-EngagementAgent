"""
Presence and Attendance Duration Tracker
Maintains cumulative presence time, session entry/exit timestamps,
and evaluates sustained attendance criteria per participant.
"""

import time
from typing import Dict, List, Optional

class ParticipantPresence:
    """Tracks continuous and cumulative presence for a single participant."""
    def __init__(self, participant_id: str, track_id: int, initial_time: float):
        self.participant_id = participant_id
        self.track_id = track_id
        self.first_seen = initial_time
        self.last_seen = initial_time
        self.total_presence_seconds = 0.0
        self.total_frames = 1
        self.is_currently_present = True
        self._last_frame_time = initial_time

    def record_frame(self, current_time: float):
        """Accumulates presence duration based on elapsed time between frames."""
        time_delta = current_time - self._last_frame_time
        # Clamp delta to prevent huge jumps from paused streams or dropped connections
        if 0 < time_delta <= 1.5:
            self.total_presence_seconds += time_delta
        else:
            self.total_presence_seconds += 0.033  # default ~30 FPS frame duration

        self.last_seen = current_time
        self._last_frame_time = current_time
        self.total_frames += 1
        self.is_currently_present = True

    def mark_absent(self):
        self.is_currently_present = False

    def to_dict(self, min_attendance_threshold_sec: float = 10.0) -> Dict:
        status = "PRESENT" if self.total_presence_seconds >= min_attendance_threshold_sec else "INSUFFICIENT_PRESENCE"
        return {
            "participant_id": self.participant_id,
            "track_id": self.track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "presence_duration_seconds": round(self.total_presence_seconds, 2),
            "total_frames_detected": self.total_frames,
            "is_currently_present": self.is_currently_present,
            "attendance_status": status,
        }

class PresenceTracker:
    """
    Orchestrates presence monitoring across all participants in a session.
    Determines sustained presence for accurate, automated attendance marking.
    """
    def __init__(self, min_attendance_threshold_sec: float = 10.0, absent_timeout_sec: float = 3.0):
        self.min_attendance_threshold_sec = min_attendance_threshold_sec
        self.absent_timeout_sec = absent_timeout_sec
        self.participants: Dict[str, ParticipantPresence] = {}
        self.session_start_time = time.time()

    def update(self, tracked_participants: List[Dict]) -> List[Dict]:
        """
        Updates presence records from the active tracker output.
        Returns a list of structured participant presence summaries.
        """
        current_time = time.time()
        active_ids = set()

        for item in tracked_participants:
            pid = item["participant_id"]
            tid = item["track_id"]
            active_ids.add(pid)

            if pid not in self.participants:
                self.participants[pid] = ParticipantPresence(pid, tid, current_time)
            else:
                self.participants[pid].record_frame(current_time)

        # Update presence status for participants not seen in this frame
        for pid, record in self.participants.items():
            if pid not in active_ids:
                if current_time - record.last_seen > self.absent_timeout_sec:
                    record.mark_absent()

        return self.get_summary()

    def get_summary(self) -> List[Dict]:
        """Returns structured attendance & presence summary for all participants."""
        return [
            p.to_dict(self.min_attendance_threshold_sec)
            for p in self.participants.values()
        ]
