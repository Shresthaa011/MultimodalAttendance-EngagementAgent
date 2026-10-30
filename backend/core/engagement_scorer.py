"""
Temporal Engagement Scoring & Profiling Module
Multimodal Attendance & Engagement Agent

Transforms fused multimodal signals (sustained visual presence, movement frequency,
inactivity duration, and speech activity intensity) into interpretable temporal engagement profiles.
Identifies attention fluctuations, peak engagement periods, and extended disengagement phases.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class EngagementScorer:
    """
    Computes transparent, adjustable engagement scores across regular time intervals,
    constructing a temporal engagement profile with trend, peak, and disengagement analytics.
    """
    PRESETS = {
        "standard": {
            "presence_weight": 0.30,
            "movement_weight": 0.20,
            "inactivity_weight": 0.20,
            "speech_weight": 0.30,
            "target_presence_sec": 10.0,
            "inactivity_limit_sec": 8.0,
            "optimal_velocity_min": 2.5,
            "optimal_velocity_max": 25.0,
            "target_speech_ratio": 0.35,
        },
        "lecture": {
            # High weight on sustained presence and calm listening, lower speech requirement
            "presence_weight": 0.45,
            "movement_weight": 0.20,
            "inactivity_weight": 0.15,
            "speech_weight": 0.20,
            "target_presence_sec": 8.0,
            "inactivity_limit_sec": 15.0,
            "optimal_velocity_min": 1.0,
            "optimal_velocity_max": 20.0,
            "target_speech_ratio": 0.15,
        },
        "discussion": {
            # High weight on vocal participation and dynamic physical interaction
            "presence_weight": 0.20,
            "movement_weight": 0.25,
            "inactivity_weight": 0.15,
            "speech_weight": 0.40,
            "target_presence_sec": 5.0,
            "inactivity_limit_sec": 6.0,
            "optimal_velocity_min": 3.0,
            "optimal_velocity_max": 30.0,
            "target_speech_ratio": 0.40,
        },
        "exam": {
            # Strict stillness and silence
            "presence_weight": 0.50,
            "movement_weight": 0.20,
            "inactivity_weight": 0.10,
            "speech_weight": 0.20,
            "target_presence_sec": 10.0,
            "inactivity_limit_sec": 30.0,
            "optimal_velocity_min": 0.5,
            "optimal_velocity_max": 10.0,
            "target_speech_ratio": 0.05,
        },
    }

    def __init__(
        self,
        mode: str = "standard",
        presence_weight: Optional[float] = None,
        movement_weight: Optional[float] = None,
        inactivity_weight: Optional[float] = None,
        speech_weight: Optional[float] = None,
    ):
        self.mode = mode if mode in self.PRESETS else "standard"
        preset = self.PRESETS[self.mode].copy()

        self.presence_weight = presence_weight if presence_weight is not None else preset["presence_weight"]
        self.movement_weight = movement_weight if movement_weight is not None else preset["movement_weight"]
        self.inactivity_weight = inactivity_weight if inactivity_weight is not None else preset["inactivity_weight"]
        self.speech_weight = speech_weight if speech_weight is not None else preset["speech_weight"]

        self.target_presence_sec = preset["target_presence_sec"]
        self.inactivity_limit_sec = preset["inactivity_limit_sec"]
        self.optimal_velocity_min = preset["optimal_velocity_min"]
        self.optimal_velocity_max = preset["optimal_velocity_max"]
        self.target_speech_ratio = preset["target_speech_ratio"]

        # Temporal Profile Sequence: list of interval records
        self.temporal_profile: List[Dict[str, Any]] = []

    def configure(
        self,
        mode: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """Allows dynamic recalibration of scoring logic and thresholds based on classroom needs."""
        if mode and mode in self.PRESETS:
            self.mode = mode
            preset = self.PRESETS[mode]
            self.presence_weight = preset["presence_weight"]
            self.movement_weight = preset["movement_weight"]
            self.inactivity_weight = preset["inactivity_weight"]
            self.speech_weight = preset["speech_weight"]
            self.target_presence_sec = preset["target_presence_sec"]
            self.inactivity_limit_sec = preset["inactivity_limit_sec"]
            self.optimal_velocity_min = preset["optimal_velocity_min"]
            self.optimal_velocity_max = preset["optimal_velocity_max"]
            self.target_speech_ratio = preset["target_speech_ratio"]

        if weights:
            self.presence_weight = weights.get("presence", self.presence_weight)
            self.movement_weight = weights.get("movement", self.movement_weight)
            self.inactivity_weight = weights.get("inactivity", self.inactivity_weight)
            self.speech_weight = weights.get("speech", self.speech_weight)

        if thresholds:
            self.target_presence_sec = thresholds.get("target_presence_sec", self.target_presence_sec)
            self.inactivity_limit_sec = thresholds.get("inactivity_limit_sec", self.inactivity_limit_sec)
            self.optimal_velocity_min = thresholds.get("optimal_velocity_min", self.optimal_velocity_min)
            self.optimal_velocity_max = thresholds.get("optimal_velocity_max", self.optimal_velocity_max)
            self.target_speech_ratio = thresholds.get("target_speech_ratio", self.target_speech_ratio)

        # Normalize weights to sum exactly to 1.0
        total_w = self.presence_weight + self.movement_weight + self.inactivity_weight + self.speech_weight
        if total_w > 0:
            self.presence_weight /= total_w
            self.movement_weight /= total_w
            self.inactivity_weight /= total_w
            self.speech_weight /= total_w

    def compute_interval_score(
        self,
        fused_data: Dict[str, Any],
        interval_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes transparent engagement score for a single time interval.
        Collectively considers presence duration, movement frequency, inactivity duration,
        and speech activity intensity.
        """
        ts = interval_timestamp or fused_data.get("sync_timestamp", time.time())
        modalities = fused_data.get("modalities", {})
        vision = modalities.get("vision", fused_data.get("vision", {}))
        audio = modalities.get("audio", fused_data.get("audio", {}))

        # ------------------------------------------------------------------
        # Factor 1: Sustained Visual Presence Score
        # ------------------------------------------------------------------
        presence_sec = float(vision.get("presence_duration_sec", 0.0))
        active_participants = int(vision.get("active_participants", 1))
        
        # Normalize presence against target presence duration
        if self.target_presence_sec > 0:
            s_presence = min(1.0, presence_sec / self.target_presence_sec)
        else:
            s_presence = 1.0 if presence_sec > 0 else 0.0

        if active_participants == 0:
            s_presence = 0.0

        # ------------------------------------------------------------------
        # Factor 2: Movement Frequency / Velocity Score
        # ------------------------------------------------------------------
        vel = float(vision.get("mean_velocity_px_sec", 0.0))
        if self.optimal_velocity_min <= vel <= self.optimal_velocity_max:
            # Optimal natural movement
            s_movement = 1.0
        elif vel < self.optimal_velocity_min:
            # Stationary / still: slightly reduced unless in lecture/exam mode
            s_movement = 0.85 if self.mode in ["lecture", "exam"] else 0.60
        else:
            # Restless / erratic excessive movement: penalized proportionally
            excess = vel - self.optimal_velocity_max
            s_movement = max(0.10, 1.0 - (excess / 30.0))

        # ------------------------------------------------------------------
        # Factor 3: Inactivity Duration Score
        # ------------------------------------------------------------------
        inactivity_sec = float(vision.get("inactivity_periods_sec", 0.0))
        if inactivity_sec <= 2.0:
            s_inactivity = 1.0
        elif inactivity_sec >= self.inactivity_limit_sec:
            s_inactivity = 0.20
        else:
            # Linear penalty between 2.0s and inactivity_limit_sec
            span = self.inactivity_limit_sec - 2.0
            s_inactivity = 1.0 - (0.80 * ((inactivity_sec - 2.0) / span))
        s_inactivity = max(0.0, min(1.0, s_inactivity))

        # ------------------------------------------------------------------
        # Factor 4: Speech Activity Intensity Score
        # ------------------------------------------------------------------
        speech_ratio = float(audio.get("speech_ratio", 0.0))
        noise_db = float(audio.get("noise_level_db", -50.0))

        if self.mode == "lecture":
            # In lecture mode, high quiet listening is desired
            s_speech = 1.0 if speech_ratio <= 0.15 and noise_db <= -40.0 else (0.80 if speech_ratio > 0.15 else 0.50)
        elif self.mode == "exam":
            # In exam mode, silence is mandatory
            s_speech = 1.0 if noise_db <= -45.0 and speech_ratio < 0.05 else 0.20
        else:
            # Standard / Discussion mode: active speech scales positively up to target
            s_speech = min(1.0, speech_ratio / max(0.01, self.target_speech_ratio))
            if speech_ratio < 0.05:
                # Moderate baseline for quiet attentiveness
                s_speech = 0.50 if noise_db <= -45.0 else 0.30

        # ------------------------------------------------------------------
        # Composite Interval Engagement Score Formulation
        # ------------------------------------------------------------------
        composite_score = (
            (self.presence_weight * s_presence) +
            (self.movement_weight * s_movement) +
            (self.inactivity_weight * s_inactivity) +
            (self.speech_weight * s_speech)
        )
        composite_score = round(max(0.0, min(1.0, composite_score)), 3)

        # Engagement Classification
        if composite_score >= 0.70:
            level = "HIGH"
        elif composite_score >= 0.45:
            level = "MODERATE"
        else:
            level = "LOW"

        interval_record = {
            "interval_index": len(self.temporal_profile),
            "timestamp": round(ts, 3),
            "engagement_score": composite_score,
            "engagement_level": level,
            "factor_sub_scores": {
                "presence_score": round(s_presence, 3),
                "movement_score": round(s_movement, 3),
                "inactivity_score": round(s_inactivity, 3),
                "speech_score": round(s_speech, 3),
            },
            "active_weights": {
                "presence_weight": round(self.presence_weight, 2),
                "movement_weight": round(self.movement_weight, 2),
                "inactivity_weight": round(self.inactivity_weight, 2),
                "speech_weight": round(self.speech_weight, 2),
            },
            "mode": self.mode,
        }

        self.temporal_profile.append(interval_record)
        return interval_record

    def analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """
        Analyzes the temporal engagement profile across the session.
        Identifies attention fluctuations, peak engagement periods, and extended disengagement phases.
        """
        if not self.temporal_profile:
            return {
                "total_intervals": 0,
                "mean_engagement_score": 0.50,
                "attention_fluctuation_index": 0.0,
                "engagement_trend": "stable",
                "trend_slope": 0.0,
                "peak_engagement_periods": [],
                "extended_disengagement_phases": [],
                "session_engagement_level": "MODERATE",
            }

        scores = [rec["engagement_score"] for rec in self.temporal_profile]
        timestamps = [rec["timestamp"] for rec in self.temporal_profile]
        mean_score = round(float(np.mean(scores)), 3)
        fluctuation = round(float(np.std(scores)), 3)  # Standard deviation = Attention Fluctuation Index

        # ------------------------------------------------------------------
        # Temporal Trend Estimation (Linear Regression Slope)
        # ------------------------------------------------------------------
        if len(scores) >= 3:
            x = np.arange(len(scores))
            slope, _ = np.polyfit(x, scores, 1)
            slope = round(float(slope), 4)
            if slope > 0.02:
                trend = "improving"
            elif slope < -0.02:
                trend = "declining"
            else:
                trend = "stable"
        else:
            slope = 0.0
            trend = "stable"

        # ------------------------------------------------------------------
        # Peak Engagement Periods (Continuous intervals where score >= 0.70)
        # ------------------------------------------------------------------
        peaks = []
        in_peak = False
        peak_start_idx = 0

        for idx, s in enumerate(scores):
            if s >= 0.70 and not in_peak:
                in_peak = True
                peak_start_idx = idx
            elif s < 0.70 and in_peak:
                in_peak = False
                peak_len = idx - peak_start_idx
                if peak_len >= 2:  # Sustained peak
                    start_t = timestamps[peak_start_idx]
                    end_t = timestamps[idx - 1]
                    peaks.append({
                        "start_interval": peak_start_idx,
                        "end_interval": idx - 1,
                        "duration_sec": round(end_t - start_t, 2) if end_t >= start_t else peak_len * 1.0,
                        "mean_score": round(float(np.mean(scores[peak_start_idx:idx])), 3),
                        "peak_type": "high_sustained_focus",
                    })

        if in_peak:
            peak_len = len(scores) - peak_start_idx
            if peak_len >= 2:
                start_t = timestamps[peak_start_idx]
                end_t = timestamps[-1]
                peaks.append({
                    "start_interval": peak_start_idx,
                    "end_interval": len(scores) - 1,
                    "duration_sec": round(end_t - start_t, 2) if end_t >= start_t else peak_len * 1.0,
                    "mean_score": round(float(np.mean(scores[peak_start_idx:])), 3),
                    "peak_type": "high_sustained_focus",
                })

        # ------------------------------------------------------------------
        # Extended Disengagement Phases (Continuous intervals where score < 0.45)
        # ------------------------------------------------------------------
        disengagements = []
        in_diseng = False
        diseng_start_idx = 0

        for idx, s in enumerate(scores):
            if s < 0.45 and not in_diseng:
                in_diseng = True
                diseng_start_idx = idx
            elif s >= 0.45 and in_diseng:
                in_diseng = False
                diseng_len = idx - diseng_start_idx
                if diseng_len >= 2:  # Extended disengagement
                    start_t = timestamps[diseng_start_idx]
                    end_t = timestamps[idx - 1]
                    disengagements.append({
                        "start_interval": diseng_start_idx,
                        "end_interval": idx - 1,
                        "duration_sec": round(end_t - start_t, 2) if end_t >= start_t else diseng_len * 1.0,
                        "mean_score": round(float(np.mean(scores[diseng_start_idx:idx])), 3),
                        "severity": "CRITICAL" if np.mean(scores[diseng_start_idx:idx]) < 0.30 else "MODERATE",
                    })

        if in_diseng:
            diseng_len = len(scores) - diseng_start_idx
            if diseng_len >= 2:
                start_t = timestamps[diseng_start_idx]
                end_t = timestamps[-1]
                disengagements.append({
                    "start_interval": diseng_start_idx,
                    "end_interval": len(scores) - 1,
                    "duration_sec": round(end_t - start_t, 2) if end_t >= start_t else diseng_len * 1.0,
                    "mean_score": round(float(np.mean(scores[diseng_start_idx:])), 3),
                    "severity": "CRITICAL" if np.mean(scores[diseng_start_idx:]) < 0.30 else "MODERATE",
                })

        # Overall Session Engagement Level
        if mean_score >= 0.70:
            overall_level = "HIGH"
        elif mean_score >= 0.45:
            overall_level = "MODERATE"
        else:
            overall_level = "LOW"

        return {
            "total_intervals": len(self.temporal_profile),
            "mean_engagement_score": mean_score,
            "session_engagement_level": overall_level,
            "attention_fluctuation_index": fluctuation,
            "engagement_trend": trend,
            "trend_slope": slope,
            "peak_engagement_periods": peaks,
            "extended_disengagement_phases": disengagements,
            "mode": self.mode,
        }

    def reset(self):
        """Resets the temporal profile history."""
        self.temporal_profile.clear()

