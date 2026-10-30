"""
LLM Insights & Pedagogical Recommendation Generator
Multimodal Attendance & Engagement Agent

Integrates Groq LLM inference with contextual multimodal session data.
Provides explainable classroom insights and actionable pedagogical recommendations.
"""

import os
from typing import Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class InsightGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key and self.api_key != "your_groq_api_key_here":
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    def _build_prompt(self, session_data: dict) -> str:
        attendance = session_data.get("attendance", {})
        visual = session_data.get("visual", {})
        audio = session_data.get("audio", {})
        multimodal = session_data.get("multimodal", {})

        present = attendance.get("present_count", 0)
        total = attendance.get("total_students", present or 1)
        att_rate = round((present / total * 100), 1) if total > 0 else 0.0

        v_score = visual.get("visual_score", visual.get("attention_score", 0.70))
        noise_db = audio.get("noise_level_db", -40.0)
        noise_stab = audio.get("noise_stability", "steady")
        speech_ratio = audio.get("speech_ratio", 0.0)
        act_pattern = audio.get("activity_pattern", "quiet_listening")
        intensity = audio.get("interaction_intensity", "silent_listening")

        mei = multimodal.get("engagement_score", 0.65)
        state = multimodal.get("primary_state", "moderate_steady_engagement")
        rules = multimodal.get("rules_triggered", ["Normal classroom baseline"])

        return (
            f"Classroom Multimodal Session Data:\n"
            f"- Attendance: {present}/{total} present ({att_rate}% verified presence)\n"
            f"- Visual Engagement Score: {v_score} (Attention index based on sustained presence)\n"
            f"- Audio Dynamics: Pattern='{act_pattern}', Noise={noise_db} dB ({noise_stab}), Speech Ratio={speech_ratio}, Intensity='{intensity}'\n"
            f"- Multimodal State: '{state}', Composite Engagement Index={mei}\n"
            f"- Reasoning Rules Triggered: {', '.join(rules)}\n\n"
            f"Based on this data, provide:\n"
            f"Insight: [1-2 concise sentences diagnosing classroom dynamic and engagement level]\n"
            f"Recommendation: [1 practical, actionable pedagogical instruction for the teacher]"
        )

    def generate_classroom_insight(self, session_data: dict) -> dict:
        prompt = self._build_prompt(session_data)

        if not self.client:
            return self._generate_fallback_insight(session_data)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational AI assistant. Provide brief, professional, and actionable insights for instructors based on multimodal student engagement data.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="mixtral-8x7b-32768",
                temperature=0.5,
                max_tokens=150,
            )

            response_text = chat_completion.choices[0].message.content

            # Simple Parsing
            insight = "Class status is stable."
            recommendation = "Maintain current pace."

            lines = response_text.strip().split("\n")
            for line in lines:
                clean = line.strip()
                if clean.lower().startswith("insight:"):
                    insight = clean.split(":", 1)[1].strip()
                elif clean.lower().startswith("recommendation:"):
                    recommendation = clean.split(":", 1)[1].strip()

            return {
                "status": "success",
                "insight": insight,
                "recommendation": recommendation,
                "source": "groq_llm",
                "model": "mixtral-8x7b-32768",
                "raw_response": response_text,
            }

        except Exception as e:
            fallback = self._generate_fallback_insight(session_data)
            fallback["error"] = str(e)
            return fallback

    def _generate_fallback_insight(self, session_data: dict) -> dict:
        """
        Deterministic, rule-based pedagogical insights when LLM is offline or during testing.
        Guarantees reliable operation in all deployment environments.
        """
        multimodal = session_data.get("multimodal", {})
        state = multimodal.get("primary_state", "moderate_steady_engagement")
        mei = multimodal.get("engagement_score", 0.65)

        if state == "active_collaboration":
            insight = "High multimodal engagement observed with dynamic student participation and sustained visual attentiveness."
            recommendation = "Facilitate structured student sharing and summarize key collaborative takeaways."
        elif state == "focused_listening":
            insight = "Classroom demonstrates calm acoustic stability and high visual focus, indicating deep lecture absorption."
            recommendation = "Maintain instructional delivery and prepare a brief checkpoint question in 5 minutes."
        elif state == "environmental_disturbance":
            insight = "Acoustic disruptions and noise instability are degrading student focus despite physical attendance."
            recommendation = "Address external noise sources or pause briefly to restore acoustic focus."
        elif state == "passive_inattention":
            insight = "Visual inattention and passive silence detected; student engagement is trending downward."
            recommendation = "Introduce an interactive poll, ask an open prompt, or initiate a brief group check-in."
        elif state == "off_camera_verbal_interaction":
            insight = "Active verbal participation detected from students seated outside direct camera focus."
            recommendation = "Acknowledge verbal contributions and encourage participants to position themselves toward shared focal materials."
        else:
            insight = "Class status is stable with balanced visual presence and acoustic metrics."
            recommendation = "Maintain current pace and continue scheduled curriculum progression."

        return {
            "status": "success",
            "insight": insight,
            "recommendation": recommendation,
            "source": "rule_based_fallback",
            "model": "rule_engine_v1",
            "composite_engagement_index": mei,
        }

