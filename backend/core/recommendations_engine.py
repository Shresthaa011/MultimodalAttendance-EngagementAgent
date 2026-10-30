"""
Pedagogical Recommendations Engine
Multimodal Attendance & Engagement Agent

Generates rule-based, non-prescriptive teaching recommendations for instructors
grounded in observable multimodal trends, temporal engagement profiles, and classroom dynamics.
"""

from typing import Dict, List, Any, Optional

class RecommendationsEngine:
    def generate_realtime_recommendations(self, session_status: dict) -> List[Dict[str, str]]:
        """
        Rule-based recommendations for immediate feedback.
        Input: Session status dict (people, emotions, attention, noise, peaks, disengagements)
        Output: List of { type: 'alert'|'warning'|'info', message: str }
        """
        recs = []

        # 1. Low Attention Alert
        avg_att = 0
        people = session_status.get('people', [])
        if people:
            avg_att = sum(p['attention'] for p in people) / len(people)

            # Individual check
            for p in people:
                if p['attention'] < 40:
                    recs.append({
                        'type': 'alert',
                        'message': f"Student {p['id']} is distracted ({p['attention']}%)."
                    })

        # 2. Global Attention Warning
        if len(people) > 2 and avg_att < 60:
            recs.append({
                'type': 'warning',
                'message': f"Global class attention is low ({round(avg_att, 1)}%). Consider changing pace or asking an interactive question."
            })

        # 3. Extended Disengagement Phase Cue (Rule-Based Pattern Detection)
        diseng_phases = session_status.get("extended_disengagement_phases", [])
        inactivity_sec = session_status.get("inactivity_periods_sec", 0.0)
        recent_trend = session_status.get("engagement_trend", "stable")

        if diseng_phases or inactivity_sec >= 8.0 or recent_trend == "declining":
            recs.append({
                'type': 'warning',
                'message': "Extended low-engagement period detected. Suggestion: Consider introducing a brief 2-minute Think-Pair-Share check-in or asking an open prompt to revitalize participation."
            })

        # 4. Peak Engagement Recognition (Highlighting Effective Teaching Segments)
        peaks = session_status.get("peak_engagement_periods", [])
        current_engagement = session_status.get("engagement_score", 0.0)
        if current_engagement <= 1.0 and current_engagement > 0:
            current_engagement *= 100.0  # Normalize to 0-100 scale if needed

        if peaks or current_engagement >= 75.0 or session_status.get("primary_state") in ["active_collaboration", "attentive_listening"]:
            recs.append({
                'type': 'info',
                'priority': 'High Priority',
                'title': 'Challenge students with harder problem',
                'impact': 'Deep Learning',
                'status_banner': 'Engagement is optimal. Keep going!',
                'message': "Effective teaching segment detected! High collective engagement and sustained student focus observed. Recommend noting this delivery method for future sessions."
            })

        # 5. Classroom Management & High Noise Intervals
        noise_db = session_status.get("noise_level_db", -50.0)
        noise_stability = session_status.get("noise_stability", "steady")
        speech_ratio = session_status.get("speech_ratio", 0.0)

        if noise_db > -30.0 and speech_ratio < 0.12 and noise_stability == "erratic_disturbance":
            recs.append({
                'type': 'warning',
                'message': f"High environmental noise level ({round(noise_db, 1)} dB) detected without active student speech. Suggestion: Check classroom acoustics or external noise sources to minimize distraction."
            })
        elif speech_ratio >= 0.30 and noise_db >= -35.0:
            recs.append({
                'type': 'info',
                'message': "Dynamic discussion active: Strong vocal participation and high student interaction observed. Maintain current discussion flow before transitioning."
            })

        return recs

    def generate_session_teaching_suggestions(
        self,
        temporal_profile: List[Dict[str, Any]],
        session_analytics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesizes session-level teaching suggestions from the temporal engagement profile.
        Outputs non-prescriptive, evidence-grounded insights for instructional improvement.
        """
        mean_score = session_analytics.get("mean_engagement_score", 0.50)
        fluctuation = session_analytics.get("attention_fluctuation_index", 0.0)
        trend = session_analytics.get("engagement_trend", "stable")
        peaks = session_analytics.get("peak_engagement_periods", [])
        disengagements = session_analytics.get("extended_disengagement_phases", [])

        teaching_suggestions = []
        positive_highlights = []
        management_cues = []

        # 1. Evaluate Peak Segments
        if peaks:
            total_peak_time = sum(p.get("duration_sec", 0.0) for p in peaks)
            positive_highlights.append(
                f"Peak Engagement: Observed {len(peaks)} high-attention period(s) totaling {round(total_peak_time, 1)}s. "
                "Teaching delivery was highly effective during these segments."
            )
        else:
            positive_highlights.append("Steady Baseline: Engagement remained consistent across the instructional timeline.")

        # 2. Evaluate Disengagement & Fatigue Drops
        if disengagements:
            total_diseng_time = sum(d.get("duration_sec", 0.0) for d in disengagements)
            teaching_suggestions.append({
                "category": "Interactivity & Pacing",
                "observation": f"{len(disengagements)} extended disengagement phase(s) detected ({round(total_diseng_time, 1)}s total).",
                "suggestion": "Introduce short interactive polls or conceptual checkpoint questions approximately 15 minutes into lecture delivery.",
            })

        # 3. Evaluate Attention Fluctuations
        if fluctuation > 0.22:
            teaching_suggestions.append({
                "category": "Attention Stabilization",
                "observation": f"Attention fluctuation index is high ({round(fluctuation, 3)} std dev).",
                "suggestion": "Incorporate predictable segment transitions and verbal signposts (e.g. 'Now shifting to our core example') to smooth attention swings.",
            })

        # 4. Overall Trend Support
        if trend == "declining":
            teaching_suggestions.append({
                "category": "Energy Restoration",
                "observation": "Engagement exhibited a downward slope toward the conclusion of the session.",
                "suggestion": "Consider scheduling a brief synthesis activity or student Q&A 5-10 minutes before class conclusion.",
            })
        elif trend == "improving":
            positive_highlights.append("Positive Trajectory: Student engagement steadily increased as the session progressed.")

        return {
            "session_summary": {
                "overall_engagement_score": mean_score,
                "attention_stability": "STABLE" if fluctuation < 0.15 else ("MODERATE_FLUCTUATION" if fluctuation < 0.25 else "HIGH_VOLATILITY"),
                "trend": trend,
                "peaks_count": len(peaks),
                "disengagement_phases_count": len(disengagements),
            },
            "positive_highlights": positive_highlights,
            "teaching_suggestions": teaching_suggestions,
            "management_cues": management_cues,
            "methodology": "Rule-based non-prescriptive recommendation engine grounded in multimodal engagement profiles.",
        }

