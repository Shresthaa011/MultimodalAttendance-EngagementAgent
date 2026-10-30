"""
Multimodal Agent Core Intelligence Layer
Multimodal Attendance & Engagement Agent

Fuses Visual Tracking/Presence with Environmental Audio Dynamics.
Applies rule-based reasoning, transparent explainability, and composite
Multimodal Engagement Index (MEI) estimation for real-world classrooms.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .llm_insights import InsightGenerator

class MultimodalAgent:
    """
    Core Intelligence Layer integrating Vision and Audio modalities.
    Applies rule-based reasoning for interpretable and explainable engagement estimation.
    """
    def __init__(
        self,
        visual_weight: float = 0.55,
        audio_weight: float = 0.45,
        groq_api_key: Optional[str] = None,
    ):
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.insight_generator = InsightGenerator(api_key=groq_api_key)
        self.session_snapshots: List[Dict[str, Any]] = []

    def evaluate_multimodal_snapshot(
        self,
        vision_data: Optional[Dict[str, Any]] = None,
        audio_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Integrates current vision and audio data, applies rule-based reasoning,
        and computes an explainable multimodal engagement score.
        """
        ts = timestamp or time.time()
        v_data = vision_data or {}
        a_data = audio_data or {}

        # ------------------------------------------------------------------
        # 1. Normalize Visual Modality Signals
        # ------------------------------------------------------------------
        # Participants count and verified attendance
        tracked_count = v_data.get("participants_count", len(v_data.get("tracked_participants", [])))
        attendance_records = v_data.get("attendance_records", v_data.get("presence_summary", {}))
        
        if isinstance(attendance_records, dict):
            present_count = sum(
                1 for r in attendance_records.values()
                if (isinstance(r, dict) and r.get("attendance_status") == "PRESENT")
                or (hasattr(r, "attendance_status") and r.attendance_status == "PRESENT")
            )
        else:
            present_count = tracked_count

        # Visual attention / presence score
        # If explicit attention score is supplied, use it; otherwise compute from presence
        if "visual_score" in v_data:
            v_score = float(v_data["visual_score"])
        elif "attention_score" in v_data:
            v_score = float(v_data["attention_score"])
        else:
            # Approximate visual focus based on tracked presence (1 or more participants)
            v_score = 0.85 if present_count > 0 else (0.50 if tracked_count > 0 else 0.10)

        v_score = round(max(0.0, min(1.0, v_score)), 3)

        # ------------------------------------------------------------------
        # 2. Normalize Acoustic Modality Signals
        # ------------------------------------------------------------------
        noise_metrics = a_data.get("noise_metrics", {})
        speech_dynamics = a_data.get("speech_dynamics", {})
        fusion_cues = a_data.get("multimodal_fusion_cues", {})

        noise_db = float(noise_metrics.get("current_noise_level_db", a_data.get("noise_level_db", -50.0)))
        noise_stability = str(noise_metrics.get("noise_stability", a_data.get("noise_stability", "steady")))
        speech_ratio = float(speech_dynamics.get("recent_speech_ratio", a_data.get("speech_ratio", 0.0)))
        speech_detected = bool(speech_dynamics.get("speech_detected", a_data.get("speech_detected", False)))
        interaction_intensity = str(speech_dynamics.get("interaction_intensity", "silent_listening"))
        turns_count = int(speech_dynamics.get("speech_bursts_count", 0))

        # Acoustic engagement score
        if "acoustic_engagement_score" in fusion_cues:
            a_score = float(fusion_cues["acoustic_engagement_score"])
        elif "acoustic_engagement_score" in a_data:
            a_score = float(a_data["acoustic_engagement_score"])
        else:
            a_score = 0.50

        a_score = round(max(0.0, min(1.0, a_score)), 3)

        # ------------------------------------------------------------------
        # 3. Rule-Based Reasoning Engine (Interpretable & Explainable)
        # ------------------------------------------------------------------
        rules_triggered = []
        delta_context = 0.0
        primary_state = "moderate_steady_engagement"
        state_label = "Steady Baseline Engagement"
        explanation = ""

        # Rule R3: Environmental Disturbance (Evaluated first to catch acoustic corruption)
        if noise_db > -30.0 and speech_ratio < 0.12 and noise_stability == "erratic_disturbance":
            rules_triggered.append({
                "rule_id": "R3_ENVIRONMENTAL_DISTURBANCE",
                "condition": "Noise > -30dB, Speech < 0.12, Erratic noise stability",
                "delta": -0.25,
                "description": "High erratic sound pressure without human speech harmonics indicates non-speech environmental acoustic disturbance.",
            })
            delta_context += -0.25
            primary_state = "environmental_disturbance"
            state_label = "Acoustic Disruption"
            explanation = "Elevated erratic environmental noise is disturbing the classroom environment, hindering student attention."

        # Rule R1: Active Collaboration / Discussion
        elif v_score >= 0.50 and (speech_ratio >= 0.25 or interaction_intensity in ["high_interaction", "moderate_participation"]):
            rules_triggered.append({
                "rule_id": "R1_ACTIVE_COLLABORATION",
                "condition": "Visual focus >= 0.50, Speech ratio >= 0.25 or active interaction",
                "delta": +0.15,
                "description": "Sustained visual presence combined with vocal interaction indicates collaborative learning or group discussion.",
            })
            delta_context += +0.15
            primary_state = "active_collaboration"
            state_label = "Interactive Discussion"
            explanation = "Participants are visibly present and actively participating in verbal exchange or answering questions."

        # Rule R2: Focused Silent Listening
        elif v_score >= 0.60 and speech_ratio < 0.12 and noise_db <= -42.0:
            rules_triggered.append({
                "rule_id": "R2_FOCUSED_LISTENING",
                "condition": "Visual focus >= 0.60, Speech ratio < 0.12, Low noise <= -42dB",
                "delta": +0.10,
                "description": "High visual attentiveness in a calm acoustic environment indicates focused lecture absorption or individual work.",
            })
            delta_context += +0.10
            primary_state = "focused_listening"
            state_label = "Attentive Listening"
            explanation = "Sustained visual attention in a quiet, stable acoustic setting indicates concentrated lecture listening."

        # Rule R4: Passive Inattention / Visual Disengagement
        elif v_score < 0.40 and (speech_ratio < 0.10 or interaction_intensity == "silent_listening"):
            rules_triggered.append({
                "rule_id": "R4_PASSIVE_INATTENTION",
                "condition": "Visual focus < 0.40, Speech ratio < 0.10, Silent environment",
                "delta": -0.20,
                "description": "Low visual attention combined with acoustic silence indicates passive inattention or fatigue.",
            })
            delta_context += -0.20
            primary_state = "passive_inattention"
            state_label = "Passive Inattention"
            explanation = "Low visual attentiveness and passive silence suggest student fatigue or disengagement from lecture content."

        # Rule R5: Off-Camera Verbal Interaction / Audio-Visual Asymmetry
        elif v_score < 0.40 and (speech_ratio >= 0.20 or interaction_intensity == "high_interaction"):
            rules_triggered.append({
                "rule_id": "R5_OFF_CAMERA_INTERACTION",
                "condition": "Visual focus < 0.40, High vocal activity >= 0.20",
                "delta": +0.05,
                "description": "Verbal participation detected while frontal visual attention is low, indicating off-camera discussion or note-taking.",
            })
            delta_context += +0.05
            primary_state = "off_camera_verbal_interaction"
            state_label = "Off-Camera Participation"
            explanation = "Active vocal interaction detected despite low camera-facing visual focus, indicating verbal contribution or peer discussion."

        # Rule R6: Default Baseline
        else:
            rules_triggered.append({
                "rule_id": "R6_BASELINE_ENGAGEMENT",
                "condition": "Balanced visual presence and standard classroom acoustics",
                "delta": 0.0,
                "description": "Classroom dynamics match standard instructional baseline with steady conditions.",
            })
            primary_state = "moderate_steady_engagement"
            state_label = "Steady Baseline Engagement"
            explanation = "Standard classroom engagement with steady acoustic background and expected student presence."

        # ------------------------------------------------------------------
        # 4. Composite Multimodal Engagement Index (MEI) Calculation
        # ------------------------------------------------------------------
        raw_mei = (self.visual_weight * v_score) + (self.audio_weight * a_score) + delta_context
        bounded_mei = round(max(0.0, min(1.0, raw_mei)), 3)

        if bounded_mei >= 0.70:
            engagement_level = "HIGH"
        elif bounded_mei >= 0.45:
            engagement_level = "MEDIUM"
        else:
            engagement_level = "LOW"

        snapshot = {
            "timestamp": round(ts, 3),
            "multimodal_engagement_index": bounded_mei,
            "engagement_level": engagement_level,
            "primary_state": primary_state,
            "state_label": state_label,
            "explanation": explanation,
            "rules_triggered": rules_triggered,
            "modality_scores": {
                "visual_score": v_score,
                "acoustic_score": a_score,
                "context_delta": round(delta_context, 2),
                "visual_weight": self.visual_weight,
                "audio_weight": self.audio_weight,
            },
            "visual_summary": {
                "tracked_participants": tracked_count,
                "verified_present": present_count,
                "visual_score": v_score,
            },
            "audio_summary": {
                "noise_level_db": noise_db,
                "noise_stability": noise_stability,
                "speech_ratio": speech_ratio,
                "speech_detected": speech_detected,
                "interaction_intensity": interaction_intensity,
                "turns_count": turns_count,
            },
        }

        self.session_snapshots.append(snapshot)
        return snapshot

    def get_session_intelligence(self, max_snapshots: int = 60) -> Dict[str, Any]:
        """
        Aggregates session snapshots, analyzes temporal engagement trends,
        and generates actionable LLM-based pedagogical insights for teachers.
        """
        if not self.session_snapshots:
            # Baseline empty structure
            empty_data = {
                "attendance": {"present_count": 0, "total_students": 0},
                "visual": {"visual_score": 0.50},
                "audio": {"noise_level_db": -50.0, "speech_ratio": 0.0, "activity_pattern": "quiet_listening"},
                "multimodal": {"engagement_score": 0.50, "primary_state": "moderate_steady_engagement"},
            }
            insights = self.insight_generator.generate_classroom_insight(empty_data)
            return {
                "session_active": False,
                "total_snapshots": 0,
                "mean_engagement_index": 0.50,
                "engagement_level": "MEDIUM",
                "engagement_trend": "stable",
                "dominant_state": "moderate_steady_engagement",
                "llm_insight": insights.get("insight", "Class session not yet initiated."),
                "pedagogical_recommendation": insights.get("recommendation", "Start camera and microphone feeds."),
                "insight_metadata": insights,
            }

        recent = self.session_snapshots[-max_snapshots:]
        scores = [s["multimodal_engagement_index"] for s in recent]
        mean_score = round(float(np.mean(scores)), 3)

        # Engagement Trend (Linear slope or delta comparison)
        if len(scores) >= 4:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            diff = second_half - first_half
            if diff > 0.06:
                trend = "improving"
            elif diff < -0.06:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Dominant state across recent snapshots
        states = [s["primary_state"] for s in recent]
        dominant_state = max(set(states), key=states.count)

        latest = recent[-1]
        session_data_for_llm = {
            "attendance": {
                "present_count": latest["visual_summary"]["verified_present"],
                "total_students": latest["visual_summary"]["tracked_participants"],
            },
            "visual": {"visual_score": latest["visual_summary"]["visual_score"]},
            "audio": {
                "noise_level_db": latest["audio_summary"]["noise_level_db"],
                "noise_stability": latest["audio_summary"]["noise_stability"],
                "speech_ratio": latest["audio_summary"]["speech_ratio"],
                "activity_pattern": latest["primary_state"],
                "interaction_intensity": latest["audio_summary"]["interaction_intensity"],
            },
            "multimodal": {
                "engagement_score": mean_score,
                "primary_state": dominant_state,
                "rules_triggered": [r["rule_id"] for r in latest["rules_triggered"]],
            },
        }

        insight_result = self.insight_generator.generate_classroom_insight(session_data_for_llm)

        level = "HIGH" if mean_score >= 0.70 else ("MEDIUM" if mean_score >= 0.45 else "LOW")

        return {
            "session_active": True,
            "total_snapshots": len(self.session_snapshots),
            "recent_snapshots_analyzed": len(recent),
            "mean_engagement_index": mean_score,
            "engagement_level": level,
            "engagement_trend": trend,
            "dominant_state": dominant_state,
            "latest_snapshot": latest,
            "llm_insight": insight_result.get("insight", "Class dynamic is stable."),
            "pedagogical_recommendation": insight_result.get("recommendation", "Maintain current instructional delivery."),
            "insight_metadata": insight_result,
        }

    def reset_session(self):
        """Resets agent session memory."""
        self.session_snapshots.clear()

