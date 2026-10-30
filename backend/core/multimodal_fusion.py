"""
Multimodal Signal Synchronization & Unified Fusion Engine
Multimodal Attendance & Engagement Agent

Synchronizes asynchronous signals from Vision Module (presence duration, movement, inactivity)
and Audio Module (noise levels, speech activity, interaction intensity) using timestamp alignment.
Constructs a unified, extensible data representation for pedagogical reasoning.
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

class MultimodalSynchronizer:
    """
    Synchronizes temporal signals across visual and auditory modalities.
    Aligns video frames with audio windows and computes correlated behavioral dynamics.
    """
    def __init__(
        self,
        sync_window_sec: float = 1.0,
        visual_weight: float = 0.55,
        audio_weight: float = 0.45,
        log_dir: str = "logs/fusion",
    ):
        self.sync_window_sec = sync_window_sec
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.log_dir = log_dir

        # Temporal Sliding Buffers
        self.vision_buffer: List[Dict[str, Any]] = []
        self.audio_buffer: List[Dict[str, Any]] = []
        self.fused_history: List[Dict[str, Any]] = []

        # Tracking state for movement patterns and inactivity calculation
        self.last_track_positions: Dict[int, Tuple[float, float, float]] = {}  # track_id -> (cx, cy, last_ts)
        self.track_inactivity_start: Dict[int, float] = {}  # track_id -> stationary_start_ts

        # Extensibility registry for future modalities (e.g. biosensors, chat logs, clickers)
        self.custom_modality_handlers: Dict[str, Callable[[float], Dict[str, Any]]] = {}

        os.makedirs(self.log_dir, exist_ok=True)

    def register_vision_snapshot(
        self,
        vision_data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a vision module frame snapshot, calculates movement velocity,
        categorizes movement patterns, and tracks inactivity periods.
        """
        ts = timestamp or time.time()
        tracked_participants = vision_data.get("tracked_participants", [])
        attendance_records = vision_data.get("attendance_records", vision_data.get("presence_summary", {}))

        # Calculate presence duration
        total_presence_sec = 0.0
        if isinstance(attendance_records, dict):
            for r in attendance_records.values():
                if isinstance(r, dict):
                    total_presence_sec += r.get("cumulative_presence_seconds", 0.0)
                elif hasattr(r, "cumulative_presence_seconds"):
                    total_presence_sec += r.cumulative_presence_seconds

        # Analyze movement patterns across tracked participants
        velocities = []
        max_inactivity_sec = 0.0

        for p in tracked_participants:
            tid = p.get("track_id")
            cx, cy = p.get("centroid", (0.0, 0.0))

            if tid in self.last_track_positions:
                prev_cx, prev_cy, prev_ts = self.last_track_positions[tid]
                dt = max(1e-3, ts - prev_ts)
                dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                velocity = dist / dt  # pixels per second
                velocities.append(velocity)

                # Inactivity detection (stationary if velocity < 3.0 px/sec)
                if velocity < 3.0:
                    if tid not in self.track_inactivity_start:
                        self.track_inactivity_start[tid] = prev_ts
                    inactivity_duration = ts - self.track_inactivity_start[tid]
                    max_inactivity_sec = max(max_inactivity_sec, inactivity_duration)
                else:
                    self.track_inactivity_start.pop(tid, None)

            self.last_track_positions[tid] = (cx, cy, ts)

        mean_velocity = round(float(np.mean(velocities)), 2) if velocities else 0.0

        # Movement pattern classification
        if mean_velocity < 2.5:
            movement_pattern = "stationary"
        elif mean_velocity <= 25.0:
            movement_pattern = "moderate_natural_movement"
        else:
            movement_pattern = "restless_excessive_movement"

        # Attention score approximation
        v_score = vision_data.get("visual_score", vision_data.get("attention_score", 0.85 if len(tracked_participants) > 0 else 0.10))

        clean_record = {
            "timestamp": round(ts, 3),
            "presence_duration_sec": round(total_presence_sec, 2),
            "active_participants": len(tracked_participants),
            "mean_velocity_px_sec": mean_velocity,
            "movement_pattern": movement_pattern,
            "inactivity_periods_sec": round(max_inactivity_sec, 2),
            "visual_attention_index": round(float(v_score), 3),
            "raw_vision": vision_data,
        }

        self.vision_buffer.append(clean_record)
        # Keep buffer bounded to recent 120 items
        if len(self.vision_buffer) > 120:
            self.vision_buffer.pop(0)

        return clean_record

    def register_audio_snapshot(
        self,
        audio_data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ingests an audio module telemetry window, extracting ambient noise levels,
        speech vs. silence ratios, and acoustic interaction intensity.
        """
        ts = timestamp or time.time()
        noise_metrics = audio_data.get("noise_metrics", {})
        speech_dynamics = audio_data.get("speech_dynamics", {})
        fusion_cues = audio_data.get("multimodal_fusion_cues", {})

        noise_db = float(noise_metrics.get("current_noise_level_db", audio_data.get("noise_level_db", -50.0)))
        noise_stability = str(noise_metrics.get("noise_stability", audio_data.get("noise_stability", "steady")))
        speech_ratio = float(speech_dynamics.get("recent_speech_ratio", audio_data.get("speech_ratio", 0.0)))
        interaction_intensity = str(speech_dynamics.get("interaction_intensity", "silent_listening"))
        activity_pattern = str(fusion_cues.get("classroom_activity_state", audio_data.get("activity_pattern", "quiet_listening")))
        turns = int(speech_dynamics.get("speech_bursts_count", audio_data.get("turns_count", 0)))
        acoustic_score = float(fusion_cues.get("acoustic_engagement_score", audio_data.get("acoustic_engagement_score", 0.50)))

        clean_record = {
            "timestamp": round(ts, 3),
            "noise_level_db": round(noise_db, 2),
            "noise_stability": noise_stability,
            "speech_ratio": round(speech_ratio, 3),
            "interaction_intensity": interaction_intensity,
            "activity_pattern": activity_pattern,
            "turns_count": turns,
            "acoustic_engagement_score": round(acoustic_score, 3),
            "raw_audio": audio_data,
        }

        self.audio_buffer.append(clean_record)
        if len(self.audio_buffer) > 120:
            self.audio_buffer.pop(0)

        return clean_record

    def register_modality_extension(self, modality_name: str, handler: Callable[[float], Dict[str, Any]]):
        """
        Registers an extension hook for future modalities (e.g., biometric wristbands, chat feeds).
        Ensures high modularity and extensibility.
        """
        self.custom_modality_handlers[modality_name] = handler

    def synchronize_and_fuse(
        self,
        sync_timestamp: Optional[float] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronizes visual and audio data within the temporal alignment window [ts - sync_window_sec, ts].
        Produces a unified, extensible multimodal data representation.
        """
        target_ts = sync_timestamp or time.time()
        window_start = target_ts - self.sync_window_sec

        # 1. Match visual snapshots within sync window
        matching_vision = [v for v in self.vision_buffer if v["timestamp"] >= window_start]
        if matching_vision:
            latest_vision = matching_vision[-1]
            presence_dur = latest_vision["presence_duration_sec"]
            participants = latest_vision["active_participants"]
            mov_pattern = latest_vision["movement_pattern"]
            vel = latest_vision["mean_velocity_px_sec"]
            inact_sec = latest_vision["inactivity_periods_sec"]
            v_score = latest_vision["visual_attention_index"]
        else:
            presence_dur = 0.0
            participants = 0
            mov_pattern = "stationary"
            vel = 0.0
            inact_sec = 0.0
            v_score = 0.50

        # 2. Match audio snapshots within sync window
        matching_audio = [a for a in self.audio_buffer if a["timestamp"] >= window_start]
        if matching_audio:
            latest_audio = matching_audio[-1]
            noise_db = latest_audio["noise_level_db"]
            noise_stab = latest_audio["noise_stability"]
            speech_rat = latest_audio["speech_ratio"]
            act_pattern = latest_audio["activity_pattern"]
            intensity = latest_audio["interaction_intensity"]
            turns = latest_audio["turns_count"]
            a_score = latest_audio["acoustic_engagement_score"]
        else:
            noise_db = -50.0
            noise_stab = "steady"
            speech_rat = 0.0
            act_pattern = "quiet_listening"
            intensity = "silent_listening"
            turns = 0
            a_score = 0.50

        # 3. Process Extensible Modality Extensions
        extensions_data = {}
        for ext_name, handler in self.custom_modality_handlers.items():
            try:
                extensions_data[ext_name] = handler(target_ts)
            except Exception as ex:
                extensions_data[ext_name] = {"error": str(ex)}

        if custom_inputs:
            extensions_data.update(custom_inputs)

        # 4. Multimodal Correlation & Concordance Analysis
        delta_mod = 0.0
        if mov_pattern in ["moderate_natural_movement", "stationary"] and speech_rat >= 0.25:
            state = "dynamic_active_presentation"
            delta_mod = +0.12
            state_desc = "Concurrent speech interaction and active physical presence indicate lively pedagogical engagement."
        elif mov_pattern == "restless_excessive_movement" and noise_db > -30.0:
            state = "restless_environmental_disruption"
            delta_mod = -0.22
            state_desc = "Excessive movement coupled with erratic ambient noise indicates in-room distraction."
        elif inact_sec > 8.0 and speech_rat < 0.10:
            state = "prolonged_passive_inactivity"
            delta_mod = -0.15
            state_desc = "Prolonged inactivity and low vocal interaction suggest lecture fatigue or disengagement."
        elif mov_pattern == "stationary" and speech_rat < 0.10 and noise_db <= -45.0:
            state = "focused_silent_study"
            delta_mod = +0.08
            state_desc = "Calm visual posture and silent acoustics indicate focused reading or note-taking."
        else:
            state = "balanced_engagement"
            delta_mod = 0.0
            state_desc = "Balanced multimodal signals aligned with standard classroom baseline."

        # Proportional engagement estimation
        prop_score = (self.visual_weight * v_score) + (self.audio_weight * a_score) + delta_mod
        prop_score = round(max(0.0, min(1.0, prop_score)), 3)

        fused_record = {
            "sync_timestamp": round(target_ts, 3),
            "sync_window_sec": self.sync_window_sec,
            "modalities": {
                "vision": {
                    "presence_duration_sec": presence_dur,
                    "active_participants": participants,
                    "movement_pattern": mov_pattern,
                    "mean_velocity_px_sec": vel,
                    "inactivity_periods_sec": inact_sec,
                    "visual_attention_index": v_score,
                },
                "audio": {
                    "noise_level_db": noise_db,
                    "noise_stability": noise_stab,
                    "speech_ratio": speech_rat,
                    "activity_pattern": act_pattern,
                    "interaction_intensity": intensity,
                    "turns_count": turns,
                    "acoustic_engagement_score": a_score,
                },
                "extensions": extensions_data,
            },
            "fused_correlation": {
                "movement_acoustic_state": state,
                "state_description": state_desc,
                "proportional_engagement_score": prop_score,
                "modality_weights": {
                    "vision": self.visual_weight,
                    "audio": self.audio_weight,
                },
                "correlation_delta": round(delta_mod, 2),
            },
        }

        self.fused_history.append(fused_record)
        return fused_record

    def export_fused_logs(self, session_id: Optional[str] = None) -> Dict[str, str]:
        """Persists synchronized multimodal telemetry to structured JSON and CSV files."""
        sid = session_id or f"fused_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        json_path = os.path.join(self.log_dir, f"{sid}.json")
        csv_path = os.path.join(self.log_dir, f"{sid}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": sid,
                "total_fused_records": len(self.fused_history),
                "fused_history": self.fused_history,
            }, f, indent=2)

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("sync_timestamp,presence_duration_sec,movement_pattern,inactivity_sec,noise_db,speech_ratio,state,proportional_score\n")
            for r in self.fused_history:
                vis = r["modalities"]["vision"]
                aud = r["modalities"]["audio"]
                corr = r["fused_correlation"]
                f.write(
                    f"{r['sync_timestamp']},{vis['presence_duration_sec']},{vis['movement_pattern']},"
                    f"{vis['inactivity_periods_sec']},{aud['noise_level_db']},{aud['speech_ratio']},"
                    f"{corr['movement_acoustic_state']},{corr['proportional_engagement_score']}\n"
                )

        return {
            "json_log": json_path,
            "csv_log": csv_path,
            "json_log_path": json_path,
            "csv_report_path": csv_path,
        }

