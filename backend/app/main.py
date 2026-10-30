"""
FastAPI Application Entry Point
Multimodal Attendance & Engagement Agent
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Ensure core package is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vision_service import VisionService
from core.audio_processor import AudioProcessor
from core.multimodal_agent import MultimodalAgent
from core.multimodal_fusion import MultimodalSynchronizer
from core.ai_suggestions import AISuggestionEngine
from core.engagement_scorer import EngagementScorer
from core.recommendations_engine import RecommendationsEngine
from core.teacher_profiles import TeacherProfileService
from core.analytics_service import AnalyticsService

app = FastAPI(
    title="Multimodal Attendance & Engagement Agent API",
    description="Real-time multimodal attendance tracking and engagement analytics platform.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Service Instances
vision_service = VisionService(camera_id=0, min_attendance_threshold_sec=10.0)
audio_processor = AudioProcessor(sample_rate=16000, window_duration_sec=1.0)
multimodal_agent = MultimodalAgent(visual_weight=0.55, audio_weight=0.45)
multimodal_synchronizer = MultimodalSynchronizer(sync_window_sec=1.0, visual_weight=0.55, audio_weight=0.45)
ai_suggestion_engine = AISuggestionEngine()
engagement_scorer = EngagementScorer(mode="standard")
recommendations_engine = RecommendationsEngine()
teacher_profile_service = TeacherProfileService()
analytics_service = AnalyticsService()

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="active",
        service="Multimodal Attendance & Engagement Agent",
        version="1.0.0",
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="Multimodal Attendance & Engagement Agent",
        version="1.0.0",
    )

@app.get("/vision/status")
async def vision_status() -> Dict[str, Any]:
    """Returns the current status of the vision pipeline and camera service."""
    return {
        "camera_running": vision_service.camera_service.is_running,
        "camera_id": vision_service.camera_service.camera_id,
        "frames_processed": vision_service.frame_count,
        "tracked_participants_count": len(vision_service.tracker.tracked_objects),
        "total_participants_registered": len(vision_service.presence_tracker.participants),
    }

@app.get("/vision/attendance")
@app.get("/attendance/records")
async def get_attendance_records() -> Dict[str, Any]:
    """Returns real-time sustained presence and attendance records for all participants."""
    records = vision_service.attendance_manager.get_attendance_summary()
    present_count = sum(1 for r in records if r["attendance_status"] == "PRESENT")
    return {
        "session_id": vision_service.attendance_manager.session_id,
        "total_participants": len(records),
        "present_count": present_count,
        "absent_count": len(records) - present_count,
        "records": records,
    }

@app.post("/attendance/export")
async def export_attendance_records() -> Dict[str, Any]:
    """Persists structured attendance logs to JSON and CSV files."""
    paths = vision_service.export_attendance()
    return {
        "status": "success",
        "message": "Structured attendance records successfully exported.",
        "log_files": paths,
    }

@app.websocket("/ws/vision")
async def websocket_vision_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint streaming live detection, tracking, and presence updates.
    """
    await websocket.accept()
    try:
        while True:
            # Receive client ping or control message
            data = await websocket.receive_text()
            summary = vision_service.presence_tracker.get_summary()
            await websocket.send_json({
                "type": "telemetry",
                "frames_processed": vision_service.frame_count,
                "participants": summary,
            })
    except WebSocketDisconnect:
        pass

# ==============================================================================
# Audio Module API Endpoints
# ==============================================================================

@app.get("/audio/status")
async def audio_status() -> Dict[str, Any]:
    """Returns the current status of the audio processing pipeline."""
    return {
        "sample_rate_hz": audio_processor.sample_rate,
        "window_duration_sec": audio_processor.window_duration_sec,
        "total_windows_processed": len(audio_processor.processed_windows),
        "buffered_samples": len(audio_processor.stream_buffer),
        "latest_metrics": audio_processor.processed_windows[-1] if audio_processor.processed_windows else None,
    }

class AudioChunkPayload(BaseModel):
    pcm_base64: Optional[str] = None
    samples: Optional[List[float]] = None
    timestamp: Optional[float] = None

@app.post("/audio/analyze-chunk")
async def analyze_audio_chunk(payload: AudioChunkPayload) -> Dict[str, Any]:
    """
    Analyzes an audio segment/chunk:
    Extracts noise levels (dB), speech activity, and acoustic engagement score
    without transcribing speech, guaranteeing full privacy.
    """
    import base64
    import numpy as np

    if payload.pcm_base64:
        raw_bytes = base64.b64decode(payload.pcm_base64)
        y = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif payload.samples:
        y = np.array(payload.samples, dtype=np.float32)
    else:
        return {"error": "Either 'pcm_base64' or 'samples' must be provided."}

    metrics = audio_processor.analyzer.analyze_audio_chunk(y)
    return {
        "timestamp": payload.timestamp or time.time(),
        "metrics": metrics,
    }

@app.get("/audio/context")
async def get_audio_context() -> Dict[str, Any]:
    """
    Returns session-level audio context metrics formatted specifically
    for fusion with visual engagement data.
    """
    return audio_processor.get_session_audio_context()

@app.post("/audio/export")
async def export_audio_logs() -> Dict[str, Any]:
    """Persists structured audio analysis logs to JSON and CSV files."""
    paths = audio_processor.save_structured_logs()
    return {
        "status": "success",
        "message": "Structured audio telemetry successfully exported.",
        "log_files": paths,
    }

# ==============================================================================
# Multimodal Agent API Endpoints
# ==============================================================================

class EvaluateSnapshotPayload(BaseModel):
    vision_data: Optional[Dict[str, Any]] = None
    audio_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

@app.get("/agent/status")
async def agent_status() -> Dict[str, Any]:
    """Returns current operational status of the Multimodal Agent intelligence layer."""
    return {
        "status": "active",
        "visual_weight": multimodal_agent.visual_weight,
        "audio_weight": multimodal_agent.audio_weight,
        "total_snapshots": len(multimodal_agent.session_snapshots),
        "latest_snapshot": multimodal_agent.session_snapshots[-1] if multimodal_agent.session_snapshots else None,
    }

@app.post("/agent/evaluate")
async def evaluate_agent_snapshot(payload: EvaluateSnapshotPayload) -> Dict[str, Any]:
    """
    Evaluates current vision + audio snapshot using rule-based reasoning,
    returning explainable engagement metrics and triggered pedagogical rules.
    """
    snapshot = multimodal_agent.evaluate_multimodal_snapshot(
        vision_data=payload.vision_data,
        audio_data=payload.audio_data,
        timestamp=payload.timestamp,
    )
    return snapshot

@app.get("/agent/session-insight")
async def get_session_insight() -> Dict[str, Any]:
    """
    Generates actionable session-level pedagogical intelligence and LLM-powered
    insights using Groq (or deterministic rule-based fallback).
    """
    return multimodal_agent.get_session_intelligence()

@app.post("/agent/reset")
async def reset_agent_session() -> Dict[str, Any]:
    """Resets Multimodal Agent session tracking memory for a new class."""
    multimodal_agent.reset_session()
    return {"status": "success", "message": "Agent session memory successfully reset."}

# ==============================================================================
# Multimodal Synchronization & AI Pedagogical Suggestion Endpoints
# ==============================================================================

class FusionSyncPayload(BaseModel):
    vision_data: Optional[Dict[str, Any]] = None
    audio_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    custom_extensions: Optional[Dict[str, Any]] = None

@app.post("/fusion/sync")
async def sync_multimodal_signals(payload: FusionSyncPayload) -> Dict[str, Any]:
    """
    Synchronizes incoming visual frames and audio chunks using timestamp alignment,
    aggregating presence duration, movement patterns, inactivity, and speech metrics.
    """
    ts = payload.timestamp or time.time()
    if payload.vision_data:
        multimodal_synchronizer.register_vision_snapshot(payload.vision_data, timestamp=ts)
    if payload.audio_data:
        multimodal_synchronizer.register_audio_snapshot(payload.audio_data, timestamp=ts)

    fused = multimodal_synchronizer.synchronize_and_fuse(
        sync_timestamp=ts,
        custom_inputs=payload.custom_extensions,
    )
    return fused

class SuggestionPayload(BaseModel):
    session_data: Optional[Dict[str, Any]] = None

@app.post("/fusion/suggest")
async def generate_ai_suggestions(payload: Optional[SuggestionPayload] = None) -> Dict[str, Any]:
    """
    Generates actionable pedagogical suggestions via Groq (llama3-8b-8192) or rule fallback
    based on the latest synchronized multimodal classroom dynamics.
    """
    data = None
    if payload and payload.session_data:
        data = payload.session_data
    elif multimodal_synchronizer.fused_history:
        data = multimodal_synchronizer.fused_history[-1]
    else:
        data = multimodal_synchronizer.synchronize_and_fuse()

    return ai_suggestion_engine.generate_suggestions(data)

@app.get("/fusion/state")
async def get_fusion_state() -> Dict[str, Any]:
    """Returns the latest unified synchronized multimodal state and buffered counts."""
    latest = multimodal_synchronizer.fused_history[-1] if multimodal_synchronizer.fused_history else None
    return {
        "sync_window_sec": multimodal_synchronizer.sync_window_sec,
        "vision_buffered": len(multimodal_synchronizer.vision_buffer),
        "audio_buffered": len(multimodal_synchronizer.audio_buffer),
        "total_fused_records": len(multimodal_synchronizer.fused_history),
        "latest_fused_state": latest,
    }

@app.post("/fusion/export")
async def export_fusion_logs() -> Dict[str, Any]:
    """Persists synchronized multimodal telemetry logs to JSON and CSV."""
    paths = multimodal_synchronizer.export_fused_logs()
    return {
        "status": "success",
        "message": "Synchronized multimodal telemetry successfully exported.",
        "log_files": paths,
    }

# ==============================================================================
# Temporal Engagement Scoring & Profile Analytics Endpoints
# ==============================================================================

class ScoreIntervalPayload(BaseModel):
    fused_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class ConfigureScorerPayload(BaseModel):
    mode: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    thresholds: Optional[Dict[str, float]] = None

@app.post("/engagement/score-interval")
async def score_engagement_interval(payload: Optional[ScoreIntervalPayload] = None) -> Dict[str, Any]:
    """
    Computes transparent engagement score for a single time interval based on
    sustained presence, movement frequency, inactivity duration, and speech intensity.
    """
    data = None
    if payload and payload.fused_data:
        data = payload.fused_data
    elif multimodal_synchronizer.fused_history:
        data = multimodal_synchronizer.fused_history[-1]
    else:
        data = multimodal_synchronizer.synchronize_and_fuse()

    ts = (payload.timestamp if payload else None) or time.time()
    return engagement_scorer.compute_interval_score(data, interval_timestamp=ts)

@app.get("/engagement/profile")
async def get_engagement_profile() -> Dict[str, Any]:
    """Returns the full sequence of temporal engagement intervals across the session."""
    return {
        "mode": engagement_scorer.mode,
        "total_intervals": len(engagement_scorer.temporal_profile),
        "temporal_profile": engagement_scorer.temporal_profile,
    }

@app.get("/engagement/analytics")
async def get_engagement_analytics() -> Dict[str, Any]:
    """
    Analyzes temporal dynamics: attention fluctuation index, trend slope,
    peak engagement periods, and extended disengagement phases.
    """
    return engagement_scorer.analyze_temporal_dynamics()

@app.post("/engagement/configure")
async def configure_engagement_scorer(payload: ConfigureScorerPayload) -> Dict[str, Any]:
    """Allows fine-tuning and recalibration of scoring logic, weights, and thresholds."""
    engagement_scorer.configure(
        mode=payload.mode,
        weights=payload.weights,
        thresholds=payload.thresholds,
    )
    return {
        "status": "success",
        "message": "Engagement scoring configuration successfully updated.",
        "mode": engagement_scorer.mode,
        "active_weights": {
            "presence": engagement_scorer.presence_weight,
            "movement": engagement_scorer.movement_weight,
            "inactivity": engagement_scorer.inactivity_weight,
            "speech": engagement_scorer.speech_weight,
        },
    }

@app.post("/engagement/reset")
async def reset_engagement_scorer() -> Dict[str, Any]:
    """Resets temporal engagement profile memory for a new session."""
    engagement_scorer.reset()
    return {"status": "success", "message": "Temporal engagement profile reset."}

# ==============================================================================
# Teaching Recommendations & Teacher Profile Endpoints
# ==============================================================================

class RealtimeRecsPayload(BaseModel):
    session_status: Optional[Dict[str, Any]] = None

@app.post("/recommendations/realtime")
async def get_realtime_recommendations(payload: Optional[RealtimeRecsPayload] = None) -> Dict[str, Any]:
    """
    Evaluates real-time session status and returns context-aware pedagogical alerts,
    warnings, and positive reinforcement cues.
    """
    status = payload.session_status if payload and payload.session_status else {}
    if not status:
        # Assemble from active singleton states
        analytics = engagement_scorer.analyze_temporal_dynamics()
        status = {
            "people": [{"id": tid, "attention": int(p.history[-1][0] % 100)} for tid, p in vision_service.tracker.tracked_objects.items()],
            "inactivity_periods_sec": multimodal_synchronizer.vision_buffer[-1]["inactivity_periods_sec"] if multimodal_synchronizer.vision_buffer else 0.0,
            "extended_disengagement_phases": analytics.get("extended_disengagement_phases", []),
            "peak_engagement_periods": analytics.get("peak_engagement_periods", []),
            "engagement_score": analytics.get("mean_engagement_score", 0.50),
            "engagement_trend": analytics.get("engagement_trend", "stable"),
            "noise_level_db": audio_processor.processed_windows[-1]["noise_level_db"] if audio_processor.processed_windows else -50.0,
            "noise_stability": audio_processor.processed_windows[-1].get("noise_stability", "steady") if audio_processor.processed_windows else "steady",
            "speech_ratio": audio_processor.processed_windows[-1]["speech_ratio"] if audio_processor.processed_windows else 0.0,
        }

    recs = recommendations_engine.generate_realtime_recommendations(status)
    return {
        "status": "success",
        "total_recommendations": len(recs),
        "recommendations": recs,
    }

@app.post("/recommendations/teaching-suggestions")
async def get_teaching_suggestions() -> Dict[str, Any]:
    """
    Generates non-prescriptive, evidence-grounded teaching suggestions from
    temporal engagement profiles and session analytics.
    """
    analytics = engagement_scorer.analyze_temporal_dynamics()
    return recommendations_engine.generate_session_teaching_suggestions(
        temporal_profile=engagement_scorer.temporal_profile,
        session_analytics=analytics,
    )

@app.get("/teacher/profile/{teacher_id}")
async def get_teacher_profile(teacher_id: str) -> Dict[str, Any]:
    """
    Returns historical teacher profile, effectiveness score, trend (+12%),
    class history, and individual student trends.
    """
    return teacher_profile_service.get_profile(teacher_id)

# ==============================================================================
# Student Analytics & Session Trends Endpoints
# ==============================================================================

class RecordMetricPayload(BaseModel):
    session_id: int
    person_id: str
    attention: float
    emotion: Optional[str] = "neutral"

@app.get("/analytics/session-trends/{session_id}")
async def get_session_trends_endpoint(session_id: int) -> Dict[str, Any]:
    """Returns aggregated attention and emotion temporal timeline for a session."""
    return analytics_service.get_session_trends(session_id)

@app.get("/analytics/student-reflective/{session_id}/{student_id}")
async def get_student_reflective_endpoint(session_id: int, student_id: str) -> Dict[str, Any]:
    """
    Returns personalized, reflective student participation summary without grades
    or peer rankings, highlighting presence duration and attention consistency.
    """
    return analytics_service.get_student_reflective_summary(session_id, student_id)

@app.post("/analytics/record-metric")
async def record_metric_endpoint(payload: RecordMetricPayload) -> Dict[str, Any]:
    """Records a granular temporal metric for a student in a session."""
    analytics_service.record_person_metric(
        session_id=payload.session_id,
        person_id=payload.person_id,
        attention=payload.attention,
        emotion=payload.emotion or "neutral",
    )
    return {"status": "success", "message": "Metric recorded successfully."}

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint receiving live audio chunks from the frontend (AudioCapture.jsx)
    and streaming back real-time acoustic metrics, session context, and multimodal fusion.
    """
    await websocket.accept()
    try:
        while True:
            # Supports both binary PCM buffer or JSON audio payload
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            latest_metrics = None
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                completed_windows = audio_processor.process_streaming_chunk(raw_bytes)
                latest_metrics = completed_windows[-1] if completed_windows else audio_processor.analyzer.analyze_audio_chunk(raw_bytes)
            elif "text" in message and message["text"]:
                import json
                try:
                    payload = json.loads(message["text"])
                    if "pcm_base64" in payload:
                        import base64
                        raw_bytes = base64.b64decode(payload["pcm_base64"])
                        completed_windows = audio_processor.process_streaming_chunk(raw_bytes)
                        latest_metrics = completed_windows[-1] if completed_windows else audio_processor.analyzer.analyze_audio_chunk(raw_bytes)
                    elif "samples" in payload:
                        samples = np.array(payload["samples"], dtype=np.float32)
                        completed_windows = audio_processor.process_streaming_chunk(samples)
                        latest_metrics = completed_windows[-1] if completed_windows else audio_processor.analyzer.analyze_audio_chunk(samples)
                except Exception as ex:
                    await websocket.send_json({"type": "error", "message": str(ex)})
                    continue

            context = audio_processor.get_session_audio_context()
            ts_now = time.time()
            multimodal_synchronizer.register_audio_snapshot(context, timestamp=ts_now)
            fused_representation = multimodal_synchronizer.synchronize_and_fuse(sync_timestamp=ts_now)
            temporal_score = engagement_scorer.compute_interval_score(fused_representation, interval_timestamp=ts_now)

            multimodal_snapshot = multimodal_agent.evaluate_multimodal_snapshot(
                vision_data={
                    "participants_count": len(vision_service.tracker.tracked_objects),
                    "attendance_records": vision_service.attendance_manager.get_attendance_summary(),
                },
                audio_data=context,
            )

            # Generate immediate recommendations
            recs = recommendations_engine.generate_realtime_recommendations({
                "people": [],
                "noise_level_db": context["noise_metrics"]["current_noise_level_db"],
                "noise_stability": context["noise_metrics"]["noise_stability"],
                "speech_ratio": context["speech_dynamics"]["recent_speech_ratio"],
                "engagement_score": temporal_score["engagement_score"],
            })

            await websocket.send_json({
                "type": "audio_metrics",
                "timestamp": round(ts_now, 3),
                "metrics": latest_metrics,
                "session_context": context,
                "multimodal_engagement": multimodal_snapshot,
                "unified_fusion": fused_representation,
                "temporal_engagement": temporal_score,
                "recommendations": recs,
            })
    except WebSocketDisconnect:
        pass
