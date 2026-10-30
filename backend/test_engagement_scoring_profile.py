"""
Temporal Engagement Scoring & Profile Analytics Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Multi-factor engagement scoring (presence duration, movement frequency, inactivity, speech).
2. Temporal engagement profile generation across sequential time intervals.
3. Attention fluctuation index, trend estimation, and stability metrics.
4. Identification of peak engagement periods and extended disengagement phases.
5. Adjustable scoring logic, custom weights, and classroom mode presets (lecture, discussion, exam).
6. FastAPI REST endpoints (/engagement/*) and WebSocket temporal telemetry streaming.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Ensure backend root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.engagement_scorer import EngagementScorer
from app.main import (
    app,
    engagement_scorer,
    score_engagement_interval,
    get_engagement_profile,
    get_engagement_analytics,
    configure_engagement_scorer,
    reset_engagement_scorer,
    websocket_audio_endpoint,
    ScoreIntervalPayload,
    ConfigureScorerPayload,
)

class MockWebSocket:
    def __init__(self, incoming):
        self.incoming = incoming
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self.incoming:
            return self.incoming.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, data):
        self.sent.append(data)

def drive(coro):
    """Executes a coroutine that only awaits immediate coroutines synchronously without OS loop overhead."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

def test_factor_scoring_and_composite_formulation():
    print("Test 1: Testing factor sub-scores & composite engagement calculation...")
    scorer = EngagementScorer(mode="standard")

    # High engagement scenario: full presence, optimal movement, zero inactivity, high speech
    fused_high = {
        "sync_timestamp": 100.0,
        "modalities": {
            "vision": {
                "presence_duration_sec": 12.0,  # >= target (10s) -> 1.0
                "active_participants": 4,
                "mean_velocity_px_sec": 12.0,   # optimal [2.5, 25.0] -> 1.0
                "inactivity_periods_sec": 0.0,  # <= 2.0s -> 1.0
            },
            "audio": {
                "speech_ratio": 0.35,           # >= target (0.35) -> 1.0
                "noise_level_db": -24.0,
            },
        },
    }

    res_high = scorer.compute_interval_score(fused_high)
    sub = res_high["factor_sub_scores"]
    assert sub["presence_score"] == 1.0
    assert sub["movement_score"] == 1.0
    assert sub["inactivity_score"] == 1.0
    assert sub["speech_score"] == 1.0
    assert res_high["engagement_score"] == 1.0
    assert res_high["engagement_level"] == "HIGH"
    print(f"  High Scenario -> Score: {res_high['engagement_score']} | Sub-scores: {sub}")

    # Low engagement scenario: absent, stationary, long inactivity, zero speech
    fused_low = {
        "sync_timestamp": 101.0,
        "modalities": {
            "vision": {
                "presence_duration_sec": 0.0,
                "active_participants": 0,
                "mean_velocity_px_sec": 0.0,
                "inactivity_periods_sec": 12.0,  # > limit (8s) -> 0.20
            },
            "audio": {
                "speech_ratio": 0.0,
                "noise_level_db": -55.0,
            },
        },
    }
    res_low = scorer.compute_interval_score(fused_low)
    assert res_low["engagement_score"] < 0.40
    assert res_low["engagement_level"] == "LOW"
    print(f"  Low Scenario -> Score: {res_low['engagement_score']} | Level: {res_low['engagement_level']}")
    print("  [PASSED] Factor sub-scores and composite formulation verified.")

def test_temporal_profile_and_attention_fluctuations():
    print("\nTest 2: Testing temporal engagement profile & attention fluctuations...")
    scorer = EngagementScorer(mode="standard")

    # Simulate 8 intervals with fluctuating dynamics
    scores_pattern = [
        (8.0, 10.0, 0.0, 0.30),  # Moderate-high
        (10.0, 15.0, 0.0, 0.35), # High
        (10.0, 12.0, 1.0, 0.40), # High
        (4.0, 2.0, 4.0, 0.10),   # Drop
        (2.0, 0.5, 7.0, 0.05),   # Low
        (1.0, 0.0, 9.0, 0.0),    # Low
        (8.0, 8.0, 0.0, 0.25),   # Recovery
        (10.0, 14.0, 0.0, 0.35), # High
    ]

    for idx, (p_dur, vel, inact, s_rat) in enumerate(scores_pattern):
        data = {
            "modalities": {
                "vision": {"presence_duration_sec": p_dur, "active_participants": 2, "mean_velocity_px_sec": vel, "inactivity_periods_sec": inact},
                "audio": {"speech_ratio": s_rat, "noise_level_db": -30.0},
            }
        }
        scorer.compute_interval_score(data, interval_timestamp=1000.0 + idx * 2.0)

    profile = scorer.temporal_profile
    assert len(profile) == 8
    print(f"  Generated {len(profile)} intervals in temporal profile.")

    analytics = scorer.analyze_temporal_dynamics()
    assert analytics["total_intervals"] == 8
    assert analytics["attention_fluctuation_index"] > 0.10  # Detectable fluctuation
    print(f"  Mean Score: {analytics['mean_engagement_score']} | Fluctuation Index (std): {analytics['attention_fluctuation_index']}")
    print(f"  Trend: {analytics['engagement_trend']} (slope: {analytics['trend_slope']})")
    print("  [PASSED] Temporal profile generation and fluctuation metrics verified.")

def test_peaks_and_extended_disengagement_detection():
    print("\nTest 3: Testing identification of peak periods & disengagement phases...")
    scorer = EngagementScorer(mode="standard")

    # Ingest 3 High intervals (Peak Period)
    for i in range(3):
        scorer.compute_interval_score({
            "modalities": {
                "vision": {"presence_duration_sec": 10.0, "active_participants": 3, "mean_velocity_px_sec": 10.0, "inactivity_periods_sec": 0.0},
                "audio": {"speech_ratio": 0.35, "noise_level_db": -25.0},
            }
        }, interval_timestamp=100.0 + i)

    # Ingest 3 Low intervals (Extended Disengagement Phase)
    for i in range(3):
        scorer.compute_interval_score({
            "modalities": {
                "vision": {"presence_duration_sec": 0.5, "active_participants": 1, "mean_velocity_px_sec": 0.0, "inactivity_periods_sec": 10.0},
                "audio": {"speech_ratio": 0.0, "noise_level_db": -55.0},
            }
        }, interval_timestamp=103.0 + i)

    analytics = scorer.analyze_temporal_dynamics()
    peaks = analytics["peak_engagement_periods"]
    diseng = analytics["extended_disengagement_phases"]

    assert len(peaks) >= 1, "Failed to identify peak engagement period"
    assert peaks[0]["mean_score"] >= 0.70
    assert len(diseng) >= 1, "Failed to identify extended disengagement phase"
    assert diseng[0]["mean_score"] < 0.45

    print(f"  Peak Engagement Detected: Intervals {peaks[0]['start_interval']} to {peaks[0]['end_interval']} (Mean: {peaks[0]['mean_score']}, Type: {peaks[0]['peak_type']})")
    print(f"  Disengagement Phase Detected: Intervals {diseng[0]['start_interval']} to {diseng[0]['end_interval']} (Mean: {diseng[0]['mean_score']}, Severity: {diseng[0]['severity']})")
    print("  [PASSED] Peak engagement periods and disengagement phases successfully detected.")

def test_classroom_presets_and_fine_tuning():
    print("\nTest 4: Testing classroom presets (lecture, discussion, exam) & custom fine-tuning...")
    scorer = EngagementScorer(mode="standard")

    # 1. Switch to lecture mode
    scorer.configure(mode="lecture")
    assert scorer.mode == "lecture"
    assert scorer.presence_weight > scorer.speech_weight
    print(f"  Lecture Mode Active -> Presence Weight: {scorer.presence_weight} | Speech Weight: {scorer.speech_weight}")

    # In lecture mode, stationary presence with quiet audio gets high score
    rec_lecture = scorer.compute_interval_score({
        "modalities": {
            "vision": {"presence_duration_sec": 8.0, "active_participants": 4, "mean_velocity_px_sec": 1.5, "inactivity_periods_sec": 2.0},
            "audio": {"speech_ratio": 0.05, "noise_level_db": -45.0},
        }
    })
    assert rec_lecture["engagement_score"] >= 0.75
    print(f"  Lecture Stillness Score: {rec_lecture['engagement_score']} ({rec_lecture['engagement_level']})")

    # 2. Switch to discussion mode
    scorer.configure(mode="discussion")
    assert scorer.mode == "discussion"
    assert scorer.speech_weight > scorer.presence_weight
    print(f"  Discussion Mode Active -> Speech Weight: {scorer.speech_weight}")

    # 3. Custom fine-tuning overrides
    scorer.configure(
        weights={"presence": 0.40, "movement": 0.20, "inactivity": 0.20, "speech": 0.20},
        thresholds={"inactivity_limit_sec": 12.0}
    )
    assert round(scorer.presence_weight + scorer.movement_weight + scorer.inactivity_weight + scorer.speech_weight, 2) == 1.0
    assert scorer.inactivity_limit_sec == 12.0
    print(f"  Custom Calibration Applied: Weights={round(scorer.presence_weight, 2)}/{round(scorer.movement_weight, 2)}/{round(scorer.inactivity_weight, 2)}/{round(scorer.speech_weight, 2)}")
    print("  [PASSED] Presets and custom calibration verified.")

def test_fastapi_endpoints_and_streaming():
    print("\nTest 5: Testing FastAPI /engagement/* routes and WebSocket streaming...")
    # 1. Reset
    reset_res = drive(reset_engagement_scorer())
    assert reset_res["status"] == "success"

    # 2. Score interval endpoint
    fused_mock = {
        "modalities": {
            "vision": {"presence_duration_sec": 10.0, "active_participants": 2, "mean_velocity_px_sec": 8.0, "inactivity_periods_sec": 0.0},
            "audio": {"speech_ratio": 0.30, "noise_level_db": -25.0},
        }
    }
    score_res = drive(score_engagement_interval(ScoreIntervalPayload(fused_data=fused_mock)))
    assert score_res["engagement_score"] >= 0.70
    print(f"  POST /engagement/score-interval -> Score: {score_res['engagement_score']} | Level: {score_res['engagement_level']}")

    # 3. Profile endpoint
    profile_res = drive(get_engagement_profile())
    assert profile_res["total_intervals"] >= 1
    print(f"  GET /engagement/profile -> Total Intervals: {profile_res['total_intervals']}")

    # 4. Analytics endpoint
    analytics_res = drive(get_engagement_analytics())
    assert "mean_engagement_score" in analytics_res
    print(f"  GET /engagement/analytics -> Mean Score: {analytics_res['mean_engagement_score']}")

    # 5. Configure endpoint
    conf_res = drive(configure_engagement_scorer(ConfigureScorerPayload(mode="lecture")))
    assert conf_res["status"] == "success"
    assert conf_res["mode"] == "lecture"
    print(f"  POST /engagement/configure -> Mode: {conf_res['mode']}")

    # 6. WebSocket streaming containing temporal_engagement
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    pcm = (0.35 * np.sin(2 * np.pi * 320 * t) * 32767.0).astype(np.int16).tobytes()
    ws = MockWebSocket([{"bytes": pcm}])
    drive(websocket_audio_endpoint(ws))
    assert len(ws.sent) >= 1
    ws_msg = ws.sent[0]
    assert "temporal_engagement" in ws_msg
    te = ws_msg["temporal_engagement"]
    print(f"  WS /ws/audio Broadcast -> Temporal Score: {te['engagement_score']} ({te['engagement_level']})")
    print("  [PASSED] FastAPI engagement routes and WebSocket streaming verified.")

def main():
    print("=" * 70)
    print(" MULTIMODAL AGENT - TEMPORAL ENGAGEMENT SCORING & PROFILING TEST SUITE")
    print("=" * 70)
    try:
        test_factor_scoring_and_composite_formulation()
        test_temporal_profile_and_attention_fluctuations()
        test_peaks_and_extended_disengagement_detection()
        test_classroom_presets_and_fine_tuning()
        test_fastapi_endpoints_and_streaming()
        print("\n" + "=" * 70)
        print("ALL TEMPORAL ENGAGEMENT SCORING TESTS PASSED (5/5).")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

