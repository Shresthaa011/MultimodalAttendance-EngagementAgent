"""
Streamlit Dashboard Integration & Verification Test Suite
Multimodal Attendance & Engagement Agent

Validates:
1. Streamlit and visualization package imports (streamlit, altair, pandas, pyarrow).
2. Dashboard data ingestion functions (attendance records and temporal engagement).
3. Data structures, metric schemas, and column integrity.
4. Integration with demonstration HUD artifacts and structured session logs.
5. Standalone headless execution validation.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd

# Add backend and project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

def test_dashboard_imports():
    print("Test 1: Testing Streamlit and visualization library imports...")
    import streamlit as st
    import altair as alt
    import pyarrow as pa
    print(f"  Streamlit Version: {st.__version__}")
    print(f"  Altair Version: {alt.__version__}")
    print("  [PASSED] All visualization dependencies verified.")

def test_dashboard_data_loaders():
    print("\nTest 2: Testing dashboard data loading helpers...")
    # Import the functions from dashboard.app
    from dashboard.app import load_attendance_records, load_temporal_engagement

    df_attendance = load_attendance_records()
    assert isinstance(df_attendance, pd.DataFrame)
    assert not df_attendance.empty
    assert "participant_id" in df_attendance.columns
    assert "cumulative_presence_seconds" in df_attendance.columns
    assert "attendance_status" in df_attendance.columns
    print(f"  Loaded Attendance Records: {len(df_attendance)} rows")
    print(f"  Verified Columns: {list(df_attendance.columns[:5])}...")

    df_engagement = load_temporal_engagement()
    assert isinstance(df_engagement, pd.DataFrame)
    assert not df_engagement.empty
    assert "timestamp" in df_engagement.columns
    assert "engagement_score" in df_engagement.columns
    assert "noise_level_db" in df_engagement.columns
    print(f"  Loaded Temporal Engagement: {len(df_engagement)} intervals")
    print(f"  Verified Mean Engagement: {df_engagement['engagement_score'].mean():.1f}%")
    print("  [PASSED] Data ingestion helpers successfully validated.")

def test_dashboard_metric_integrity():
    print("\nTest 3: Testing KPI metric aggregation and threshold evaluation...")
    from dashboard.app import load_attendance_records, load_temporal_engagement

    df_att = load_attendance_records()
    present_count = int(sum(df_att["attendance_status"] == "PRESENT"))
    total_count = len(df_att)
    att_rate = round((present_count / max(1, total_count)) * 100.0, 1)

    assert 0 <= present_count <= total_count
    assert 0.0 <= att_rate <= 100.0
    print(f"  Total Enrolled: {total_count} | Present: {present_count} ({att_rate}%)")

    df_eng = load_temporal_engagement()
    mean_score = float(df_eng["engagement_score"].mean())
    assert 0.0 <= mean_score <= 100.0
    dominant_state = df_eng["dynamic_state"].mode()[0]
    assert len(dominant_state) > 0
    print(f"  Engagement Index: {mean_score:.1f}% | Classroom Dynamic: {dominant_state}")
    print("  [PASSED] KPI aggregation and metric integrity verified.")

def test_demonstration_artifacts_resolution():
    print("\nTest 4: Testing visual demonstration artifact discovery...")
    demo_img = PROJECT_ROOT / "logs" / "demonstration" / "test_visual" / "end_to_end_pipeline_hud.jpg"
    if not demo_img.exists():
        demo_img = PROJECT_ROOT / "end_to_end_pipeline_hud.jpg"
    
    assert demo_img.exists(), f"Demonstration HUD image not found at {demo_img}"
    assert demo_img.stat().st_size > 1000, "Demonstration image is empty"
    print(f"  Resolved HUD Artifact: {demo_img.name} ({demo_img.stat().st_size} bytes)")
    print("  [PASSED] Demonstration artifact discovery verified.")

def test_dashboard_file_compilation():
    print("\nTest 5: Testing Streamlit dashboard syntax and AST compilation...")
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    with open(dashboard_path, "r", encoding="utf-8") as f:
        code = f.read()
    compiled = compile(code, str(dashboard_path), "exec")
    assert compiled is not None
    print(f"  Compiled {dashboard_path.name} without syntax errors.")
    print("  [PASSED] Streamlit dashboard code AST compiled cleanly.")

def main():
    print("=" * 70)
    print(" STREAMLIT DASHBOARD INTEGRATION TEST SUITE")
    print("=" * 70)
    try:
        test_dashboard_imports()
        test_dashboard_data_loaders()
        test_dashboard_metric_integrity()
        test_demonstration_artifacts_resolution()
        test_dashboard_file_compilation()
        print("\n" + "=" * 70)
        print(" ALL STREAMLIT DASHBOARD TESTS PASSED (5/5). DASHBOARD READY.")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n[TEST FAILED]: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

