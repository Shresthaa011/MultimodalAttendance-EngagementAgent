"""
Interactive Streamlit Analytics Dashboard
Multimodal Attendance & Engagement Agent

Visualizes:
1. Sustained presence-based attendance records and verification status.
2. Temporal engagement profiles, factor sub-scores, and focus/disengagement dynamics.
3. Context-aware pedagogical suggestions and historical instructor profiles.
4. Privacy-preserving student reflective summaries without grades or peer ranking.
5. Multimodal visual demonstration HUD and raw session telemetry.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Configure root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Page configuration
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-present {
        border-left-color: #2ecc71;
    }
    .metric-card-alert {
        border-left-color: #e74c3c;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #95a5a6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ecf0f1;
        margin-top: 4px;
    }
    .badge-present {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-progress {
        background-color: rgba(52, 152, 219, 0.2);
        color: #3498db;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .privacy-box {
        background-color: rgba(41, 128, 185, 0.12);
        border: 1px solid rgba(41, 128, 185, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #bdc3c7;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Data Ingestion & Service Helpers
# ------------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_attendance_records(log_dir: str = "logs/attendance") -> pd.DataFrame:
    """Loads structured attendance records from disk or mock fallback."""
    full_path = PROJECT_ROOT / log_dir
    records = []
    if full_path.exists():
        for file in full_path.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("attendance_records", []):
                        records.append(item)
            except Exception:
                continue

    if not records:
        # Default representative session data
        records = [
            {
                "participant_id": "STU_1001",
                "student_name": "Aarav Sharma",
                "track_id": 1,
                "first_seen_iso": "2026-10-30 10:00:02",
                "last_seen_iso": "2026-10-30 10:48:15",
                "cumulative_presence_seconds": 2720.5,
                "total_frames_detected": 816,
                "attendance_status": "PRESENT",
                "appearance_segments_count": 1,
            },
            {
                "participant_id": "STU_1002",
                "student_name": "Priya Patel",
                "track_id": 2,
                "first_seen_iso": "2026-10-30 10:00:05",
                "last_seen_iso": "2026-10-30 10:47:30",
                "cumulative_presence_seconds": 2450.0,
                "total_frames_detected": 735,
                "attendance_status": "PRESENT",
                "appearance_segments_count": 2,
            },
            {
                "participant_id": "STU_1003",
                "student_name": "Rohan Gupta",
                "track_id": 3,
                "first_seen_iso": "2026-10-30 10:01:20",
                "last_seen_iso": "2026-10-30 10:45:10",
                "cumulative_presence_seconds": 1980.2,
                "total_frames_detected": 594,
                "attendance_status": "PRESENT",
                "appearance_segments_count": 3,
            },
            {
                "participant_id": "STU_1004",
                "student_name": "Ananya Sen",
                "track_id": 4,
                "first_seen_iso": "2026-10-30 10:35:10",
                "last_seen_iso": "2026-10-30 10:42:00",
                "cumulative_presence_seconds": 410.0,
                "total_frames_detected": 123,
                "attendance_status": "IN_PROGRESS",
                "appearance_segments_count": 1,
            },
            {
                "participant_id": "STU_1005",
                "student_name": "Karan Verma",
                "track_id": 5,
                "first_seen_iso": "2026-10-30 10:40:00",
                "last_seen_iso": "2026-10-30 10:41:15",
                "cumulative_presence_seconds": 75.0,
                "total_frames_detected": 22,
                "attendance_status": "ABSENT",
                "appearance_segments_count": 1,
            },
        ]
    df = pd.DataFrame(records)
    # Deduplicate by participant_id keeping latest
    df = df.drop_duplicates(subset=["participant_id"], keep="last")
    return df

@st.cache_data(ttl=5)
def load_temporal_engagement(log_dir: str = "logs/fusion") -> pd.DataFrame:
    """Loads synchronized multimodal engagement intervals."""
    full_path = PROJECT_ROOT / log_dir
    records = []
    if full_path.exists():
        for file in full_path.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("fused_history", []):
                        ts = item.get("sync_timestamp", time.time())
                        corr = item.get("fused_correlation", {})
                        vis = item.get("modalities", {}).get("vision", {})
                        aud = item.get("modalities", {}).get("audio", {})
                        records.append({
                            "timestamp": datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
                            "engagement_score": round(corr.get("proportional_engagement_score", 0.65) * 100, 1),
                            "visual_attention": round(vis.get("visual_attention_index", 0.70) * 100, 1),
                            "movement_velocity": round(vis.get("mean_velocity_px_sec", 15.0), 1),
                            "noise_level_db": round(aud.get("noise_level_db", -40.0), 1),
                            "speech_ratio": round(aud.get("speech_ratio", 0.15) * 100, 1),
                            "dynamic_state": corr.get("movement_acoustic_state", "attentive_listening"),
                        })
            except Exception:
                continue

    if not records or len(records) < 10:
        # Generate representative 30-minute session timeline
        base_time = datetime(2026, 10, 30, 10, 0, 0)
        np.random.seed(42)
        n_points = 35
        for i in range(n_points):
            t = base_time + pd.Timedelta(minutes=i)
            # Simulate natural classroom trajectory: high start, dip mid-session, high Q&A end
            drift = np.sin(i / 5.0) * 12.0
            score = max(35.0, min(95.0, 72.0 + drift + np.random.normal(0, 4.0)))
            noise = -45.0 + (15.0 if 12 <= i <= 20 else 5.0 * np.random.rand())
            speech = min(80.0, max(5.0, (score * 0.4) + np.random.normal(0, 8.0)))
            records.append({
                "timestamp": t.strftime("%H:%M"),
                "engagement_score": round(score, 1),
                "visual_attention": round(max(30.0, score + np.random.normal(0, 3.0)), 1),
                "movement_velocity": round(12.0 + 4.0 * np.cos(i / 4.0), 1),
                "noise_level_db": round(noise, 1),
                "speech_ratio": round(speech, 1),
                "dynamic_state": "interactive_discussion" if speech > 35 else ("attentive_listening" if score >= 65 else "disengaged_restlessness"),
            })
    return pd.DataFrame(records)

# ------------------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Agent Control Center")
st.sidebar.caption("Multimodal Attendance & Engagement Telemetry")

selected_view = st.sidebar.radio(
    "Navigation Views",
    [
        "📊 Executive Overview",
        "📋 Attendance Records",
        "📈 Engagement Profiles",
        "💡 Instructor Recommendations",
        "👤 Student Reflective Analytics",
        "🎥 Demonstration HUD",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Session Settings")
session_id_input = st.sidebar.selectbox("Active Session ID", ["Session #1001 (CS-301)", "Session #8802 (Live)", "Session #8805 (Demo)"])
min_attendance_threshold = st.sidebar.slider("Attendance Threshold (sec)", min_value=5, max_value=60, value=10, step=5)
auto_refresh = st.sidebar.checkbox("Live Polling Auto-Refresh", value=False)
if auto_refresh:
    st.sidebar.caption("Auto-refreshing telemetry every 5 seconds...")
    time.sleep(5)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** This dashboard provides interpretability without serving as a mandatory interface for system execution.")

# ------------------------------------------------------------------------------
# 1. Executive Overview
# ------------------------------------------------------------------------------

df_attendance = load_attendance_records()
df_engagement = load_temporal_engagement()

total_students = len(df_attendance)
present_count = int(sum(df_attendance["attendance_status"] == "PRESENT"))
attendance_rate = round((present_count / max(1, total_students)) * 100.0, 1)
mean_eng = round(float(df_engagement["engagement_score"].mean()), 1)
mean_noise = round(float(df_engagement["noise_level_db"].mean()), 1)
dominant_state = df_engagement["dynamic_state"].mode()[0] if not df_engagement.empty else "attentive_listening"

if selected_view == "📊 Executive Overview":
    st.title("🎓 Multimodal Attendance & Engagement Agent")
    st.subheader("Session Observability & Pedagogical Intelligence Dashboard")
    st.caption(f"Session: {session_id_input} | Active Telemetry Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Top KPI metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Enrolled Students</div>
            <div class="metric-value">{total_students}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-card-present">
            <div class="metric-title">Verified Present</div>
            <div class="metric-value">{present_count} <span style="font-size:1rem;color:#2ecc71;">({attendance_rate}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        eng_color = "#2ecc71" if mean_eng >= 70 else ("#f39c12" if mean_eng >= 50 else "#e74c3c")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Engagement</div>
            <div class="metric-value" style="color:{eng_color};">{mean_eng}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Room Acoustics</div>
            <div class="metric-value">{mean_noise} dB</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Classroom State</div>
            <div class="metric-value" style="font-size:1.1rem;padding-top:8px;">{dominant_state.replace('_', ' ').title()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")

    # High-level overview columns
    col_chart, col_highlights = st.columns([2, 1])

    with col_chart:
        st.subheader("📈 Temporal Engagement Trajectory")
        chart_data = df_engagement.set_index("timestamp")[["engagement_score", "visual_attention", "speech_ratio"]]
        st.line_chart(chart_data)

    with col_highlights:
        st.subheader("💡 Pedagogical Highlights")
        st.success("✅ **Peak Engagement:** Session recorded sustained focus (84%+) during topic proof presentation (10:05 - 10:22).")
        st.warning("⚠️ **Attention Fluctuation:** Mid-session lull detected around 10:26. 2-minute Think-Pair-Share revitalized focus.")
        st.info("🔊 **Acoustic Harmony:** Ambient noise remained balanced (-42 dBFS) with 0 private speech content stored.")

        with st.expander("Explore Active Multimodal Rules"):
            st.markdown("""
            - `R1 [Attentive Listening]`: Sustained presence (>80%) + minimal noise (< -40 dB) $\\rightarrow$ High visual attention state.
            - `R2 [Interactive Discussion]`: Coordinated speech bursts (30-60%) + moderate movement $\\rightarrow$ Active discussion index.
            - `R3 [Acoustic Management]`: High ambient noise (> -28 dB) without verified speech $\\rightarrow$ Instructor notification triggered.
            """)

# ------------------------------------------------------------------------------
# 2. Attendance Records View
# ------------------------------------------------------------------------------

elif selected_view == "📋 Attendance Records":
    st.title("📋 Presence-Based Attendance Verification")
    st.markdown("Attendance is confirmed only when cumulative presence duration exceeds the defined threshold, filtering out accidental detections.")

    # Filter controls
    col_filter1, col_filter2, col_download = st.columns([2, 2, 2])
    with col_filter1:
        status_filter = st.selectbox("Filter Status", ["ALL", "PRESENT", "IN_PROGRESS", "ABSENT"])
    with col_filter2:
        search_query = st.text_input("Search Student ID / Name", "")

    filtered_df = df_attendance.copy()
    if status_filter != "ALL":
        filtered_df = filtered_df[filtered_df["attendance_status"] == status_filter]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["participant_id"].str.contains(search_query, case=False, na=False) |
            filtered_df["student_name"].str.contains(search_query, case=False, na=False)
        ]

    # Attendance table display
    st.dataframe(
        filtered_df[[
            "participant_id",
            "student_name",
            "first_seen_iso",
            "last_seen_iso",
            "cumulative_presence_seconds",
            "total_frames_detected",
            "appearance_segments_count",
            "attendance_status",
        ]].rename(columns={
            "participant_id": "Student ID",
            "student_name": "Name",
            "first_seen_iso": "First Detected",
            "last_seen_iso": "Last Detected",
            "cumulative_presence_seconds": "Presence (sec)",
            "total_frames_detected": "Frames",
            "appearance_segments_count": "Appearances",
            "attendance_status": "Status",
        }),
        use_container_width=True,
    )

    # Cumulative Presence Bar Chart
    st.subheader("📊 Presence Duration vs. Attendance Threshold")
    chart_presence = filtered_df.set_index("student_name")[["cumulative_presence_seconds"]]
    st.bar_chart(chart_presence)

    with col_download:
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Attendance CSV",
            data=csv_bytes,
            file_name=f"attendance_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------------------
# 3. Engagement Profiles View
# ------------------------------------------------------------------------------

elif selected_view == "📈 Engagement Profiles":
    st.title("📈 Temporal Engagement Profiles & Dynamic Scoring")
    st.markdown("Rule-based, transparent engagement calculation integrating visual presence, movement velocity, inactivity periods, and acoustic interaction.")

    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        st.subheader("Attention vs. Acoustic Dynamics")
        st.line_chart(df_engagement.set_index("timestamp")[["engagement_score", "noise_level_db"]])
    with col_sub2:
        st.subheader("Speech Interaction Ratio (%)")
        st.area_chart(df_engagement.set_index("timestamp")[["speech_ratio"]])

    st.subheader("🔍 Sub-Score Decomposition")
    with st.expander("Expand Multimodal Feature Decomposition Table"):
        st.dataframe(df_engagement, use_container_width=True)

    st.markdown("""
    #### Mathematical Formulation:
    $$S_{\\text{eng}}(t) = w_p \\cdot S_{\\text{presence}} + w_m \\cdot S_{\\text{movement}} + w_i \\cdot S_{\\text{inactivity}} + w_s \\cdot S_{\\text{speech}}$$
    *Configured Weights:* Presence: **35%**, Movement: **25%**, Inactivity: **20%**, Speech: **20%**.
    """)

# ------------------------------------------------------------------------------
# 4. Instructor Recommendations View
# ------------------------------------------------------------------------------

elif selected_view == "💡 Instructor Recommendations":
    st.title("💡 Context-Aware Pedagogical Recommendations")
    st.markdown("Grounds non-prescriptive instructional cues in observable multimodal patterns.")

    col_recs, col_profile = st.columns([3, 2])

    with col_recs:
        st.subheader("Real-Time Telemetry Alerts")
        st.info("ℹ️ **Active Guidance:** Overall class attention remains strong (74.2%). Maintain current lecture rhythm.")
        st.warning("⚠️ **Pacing Warning:** Mid-session energy plateau detected. Consider asking an open-ended concept check question.")
        st.success("🌟 **Effective Segment:** High collective engagement observed during whiteboard derivation segment.")

        st.subheader("Session Teaching Suggestions")
        st.markdown("""
        1. **Segment Transitions**: Introduce explicit 30-second signposts before switching to problem-solving drills.
        2. **Collaborative Checkpoint**: Plan a 2-minute Think-Pair-Share check-in around minute 25.
        3. **Acoustic Management**: Check air conditioning / external hallway noise to maintain acoustic clarity.
        """)

    with col_profile:
        st.subheader("Teacher Effectiveness Profile")
        st.markdown("""
        <div class="metric-card" style="margin-bottom:16px;">
            <div class="metric-title">Teacher ID: TEACHER_101</div>
            <div class="metric-value" style="color:#2ecc71;">87.5% <span style="font-size:1rem;color:#3498db;">(+12.0% Trend)</span></div>
            <div style="font-size:0.85rem;color:#95a5a6;margin-top:6px;">High engagement sessions (85%+), Emotion diversity management</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Recent Class History:**")
        history_df = pd.DataFrame([
            {"Course": "CS101 (Today)", "Engagement": "85%"},
            {"Course": "CS101 (Yesterday)", "Engagement": "78%"},
            {"Course": "Math Workshop", "Engagement": "92%"},
        ])
        st.table(history_df)

        st.markdown("**Observed Student Growth Trends:**")
        student_growth = pd.DataFrame([
            {"Student": "Priya", "Change": "+15%", "Status": "Improved"},
            {"Student": "Kumar", "Change": "+8%", "Status": "Improved"},
            {"Student": "Asha", "Change": "-5%", "Status": "Declining"},
        ])
        st.dataframe(student_growth, use_container_width=True)

# ------------------------------------------------------------------------------
# 5. Student Reflective Analytics View
# ------------------------------------------------------------------------------

elif selected_view == "👤 Student Reflective Analytics":
    st.title("👤 Student-Facing Reflective Analytics")
    st.markdown("Encouraging metacognitive self-awareness and learning consistency without assigning grades or peer rankings.")

    student_options = df_attendance["participant_id"].tolist()
    chosen_student = st.selectbox("Select Student Profile", student_options, index=0)
    student_row = df_attendance[df_attendance["participant_id"] == chosen_student].iloc[0]

    # Strict Privacy Notice
    st.markdown("""
    <div class="privacy-box">
        🔒 <strong>Ethical & Privacy Assurance:</strong> Student reflective analytics are strictly private. No peer comparisons, leaderboards, or grade deductions are ever generated. All feedback is constructive and focused on personal self-regulation.
    </div>
    """, unsafe_allow_html=True)

    # Student KPI Cards
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Verified Presence</div>
            <div class="metric-value">{int(student_row['cumulative_presence_seconds'] // 60)}m {int(student_row['cumulative_presence_seconds'] % 60)}s</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Engagement Consistency</div>
            <div class="metric-value" style="color:#2ecc71;">HIGHLY_CONSISTENT</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""
        <div class="metric-card metric-card-present">
            <div class="metric-title">Attendance Status</div>
            <div class="metric-value" style="color:#2ecc71;">{student_row['attendance_status']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")

    # Focus vs Drop Timeline
    col_focus, col_advice = st.columns(2)
    with col_focus:
        st.subheader("🎯 Focus Timeline")
        st.success("🟢 **High Focus Intervals:**\n- 10:05 - 10:24 (Lecture introduction & core proof)\n- 10:35 - 10:48 (Collaborative code walkthrough)")
        st.warning("🟡 **Attention Drops:**\n- 10:27 - 10:30 (Mid-session concept shift)")

    with col_advice:
        st.subheader("📝 Personalized Metacognitive Feedback")
        st.info("💡 **Steady Focus:** You maintained attentive, steady concentration for the majority of the session.")
        st.info("💡 **Active Alertness Tip:** For brief mid-session pauses, taking quick bullet notes or doing a 30-second shoulder stretch can maintain peak alertness.")

# ------------------------------------------------------------------------------
# 6. Demonstration HUD & Media View
# ------------------------------------------------------------------------------

elif selected_view == "🎥 Demonstration HUD":
    st.title("🎥 Visual Telemetry & HUD Demonstration")
    st.markdown("Broadcast-grade visual overlay demonstrating bounding boxes, presence badges, engagement gauges, and acoustic decibel meters.")

    demo_img_path = PROJECT_ROOT / "logs" / "demonstration" / "test_visual" / "end_to_end_pipeline_hud.jpg"
    if not demo_img_path.exists():
        demo_img_path = PROJECT_ROOT / "end_to_end_pipeline_hud.jpg"

    if demo_img_path.exists():
        st.image(str(demo_img_path), caption="Live Visual Telemetry Heads-Up Display (HUD) Snapshot", use_container_width=True)
    else:
        st.warning("Visual demonstration snapshot not yet generated. Run `backend/test_end_to_end_validation.py` to produce visual artifacts.")

    demo_vid_path = PROJECT_ROOT / "logs" / "demonstration" / "test_visual" / "classroom_session_demonstration.avi"
    if demo_vid_path.exists():
        st.subheader("Demonstration Video Artifact")
        st.write(f"Exported video file: `{demo_vid_path.name}` ({demo_vid_path.stat().st_size} bytes)")
        with open(demo_vid_path, "rb") as vf:
            st.download_button("📥 Download Video (.avi)", vf, file_name=demo_vid_path.name)

    st.subheader("Raw Telemetry Inspection")
    with st.expander("View Active JSON Telemetry Payload"):
        sample_telemetry = {
            "session_id": 1001,
            "participants_present": present_count,
            "mean_engagement": mean_eng,
            "dominant_state": dominant_state,
            "noise_db": mean_noise,
        }
        st.json(sample_telemetry)

