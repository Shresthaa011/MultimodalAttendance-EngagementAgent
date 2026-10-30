"""
AI Pedagogical Suggestion Engine
Multimodal Attendance & Engagement Agent

Generates actionable pedagogical recommendations using Groq (llama3-8b-8192)
based on synchronized multimodal indicators (presence duration, movement, inactivity,
speech activity, and ambient noise levels).
"""

import os
from typing import Dict, Any, Optional, List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class AISuggestionEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key and self.api_key != "your_groq_api_key_here":
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    def generate_suggestions(self, session_data: dict):
        """
        Generates actionable pedagogical suggestions from synchronized session data.
        Integrates visual movement, inactivity, presence duration, and acoustic interaction.
        """
        vision = session_data.get("vision", session_data.get("modalities", {}).get("vision", {}))
        audio = session_data.get("audio", session_data.get("modalities", {}).get("audio", {}))
        correlation = session_data.get("fused_correlation", session_data.get("multimodal", {}))

        prompt = (
            f"You are an expert AI teaching assistant. Analyze this synchronized multimodal classroom telemetry:\n"
            f"- Visual Presence: {vision.get('presence_duration_sec', 0.0)}s cumulative duration, {vision.get('active_participants', 0)} active participants\n"
            f"- Visual Movement: '{vision.get('movement_pattern', 'moderate')}' (velocity: {vision.get('mean_velocity_px_sec', 0.0)} px/s), Inactivity: {vision.get('inactivity_periods_sec', 0.0)}s\n"
            f"- Audio Dynamics: Noise={audio.get('noise_level_db', -40.0)} dB, Speech Ratio={audio.get('speech_ratio', 0.0)}, Pattern='{audio.get('activity_pattern', 'quiet_listening')}'\n"
            f"- Correlation State: '{correlation.get('movement_acoustic_state', 'balanced')}' (Engagement Score: {correlation.get('proportional_engagement_score', 0.65)})\n\n"
            f"Provide structured recommendations formatted as:\n"
            f"Primary Action: [Immediate pedagogical intervention for the instructor]\n"
            f"Class Dynamic: [Diagnosis of student engagement and classroom flow]\n"
            f"Suggested Activity: [Practical pedagogical exercise or adjustment]"
        )

        if not self.client:
            return self._fallback_suggestions(session_data)

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert AI teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=200
            )

            response = completion.choices[0].message.content
            return self._parse_response(response)

        except Exception as e:
            print(f"AI Suggestion Error: {e}")
            return self._fallback_suggestions(session_data)

    def _parse_response(self, response: str) -> dict:
        """Parses the LLM generated text into structured suggestion fields."""
        primary_action = "Maintain current instructional rhythm."
        class_dynamic = "Classroom dynamics are stable."
        suggested_activity = "Continue curriculum schedule."

        for line in response.strip().split("\n"):
            clean = line.strip()
            if clean.lower().startswith("primary action:"):
                primary_action = clean.split(":", 1)[1].strip()
            elif clean.lower().startswith("class dynamic:"):
                class_dynamic = clean.split(":", 1)[1].strip()
            elif clean.lower().startswith("suggested activity:"):
                suggested_activity = clean.split(":", 1)[1].strip()

        return {
            "status": "success",
            "source": "groq_llama3",
            "model": "llama3-8b-8192",
            "primary_action": primary_action,
            "class_dynamic": class_dynamic,
            "suggested_activity": suggested_activity,
            "raw_response": response,
        }

    def _fallback_suggestions(self, session_data: dict) -> dict:
        """
        Deterministic, rule-based pedagogical suggestions when LLM is offline.
        Aligns directly with synchronized movement, inactivity, and acoustic indicators.
        """
        vision = session_data.get("vision", session_data.get("modalities", {}).get("vision", {}))
        audio = session_data.get("audio", session_data.get("modalities", {}).get("audio", {}))
        correlation = session_data.get("fused_correlation", session_data.get("multimodal", {}))

        mov_pattern = vision.get("movement_pattern", "moderate_natural_movement")
        inactivity_sec = vision.get("inactivity_periods_sec", 0.0)
        speech_ratio = audio.get("speech_ratio", 0.0)
        noise_db = audio.get("noise_level_db", -40.0)
        state = correlation.get("movement_acoustic_state", "balanced_engagement")

        if state == "dynamic_active_presentation" or (mov_pattern == "moderate_natural_movement" and speech_ratio >= 0.25):
            primary_action = "Facilitate peer questions and invite audience responses."
            class_dynamic = "High engagement with concurrent speech activity and active physical presence."
            suggested_activity = "2-minute Think-Pair-Share check-in on the current topic."
        elif state == "restless_environmental_disruption" or (mov_pattern == "restless_excessive_movement" and noise_db > -30.0):
            primary_action = "Direct class attention back to primary visual slides and pause for focus."
            class_dynamic = "Elevated movement and ambient noise indicate classroom distraction."
            suggested_activity = "Brief acoustic reset: ask for silent individual reflection on the prompt."
        elif state == "prolonged_passive_inactivity" or inactivity_sec > 8.0 or (speech_ratio < 0.08 and mov_pattern == "stationary"):
            primary_action = "Pose an interactive question or conduct a quick thumbs-up poll."
            class_dynamic = "Prolonged inactivity and low vocal interaction indicate passive lecture fatigue."
            suggested_activity = "Rapid formative check: 60-second quiz question on whiteboard."
        elif speech_ratio < 0.10 and noise_db <= -45.0:
            primary_action = "Maintain steady instructional pace; students are focused."
            class_dynamic = "Quiet, attentive listening state with steady visual posture."
            suggested_activity = "Continue topic explanation, transition to exercise in 5 minutes."
        else:
            primary_action = "Maintain current instructional pace and monitor student responses."
            class_dynamic = "Balanced classroom engagement across visual and acoustic metrics."
            suggested_activity = "Continue lecture according to syllabus schedule."

        return {
            "status": "success",
            "source": "rule_based_fallback",
            "model": "rule_engine_v1",
            "primary_action": primary_action,
            "class_dynamic": class_dynamic,
            "suggested_activity": suggested_activity,
        }

