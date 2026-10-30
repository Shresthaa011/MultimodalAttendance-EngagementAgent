# Milestone 5: Student-Facing Analytics & Reflective Session Summaries

## 1. Overview & Objectives

In educational technology, feedback mechanisms frequently err on the side of punitive evaluation, using rigid rankings, percentile comparisons, or arbitrary letter grades. Such approaches often induce evaluation anxiety rather than fostering metacognitive awareness and active self-regulation.

This activity establishes a **student-facing, reflective analytics engine** within the Multimodal Attendance & Engagement Agent. Using verified presence timestamps, continuous gaze/movement attention markers, and synchronized classroom telemetry, the system synthesizes individual participation summaries designed strictly for **self-reflection and learning improvement**.

```mermaid
flowchart TD
    subgraph Data Sources
        PR[Presence Records]
        PM[Person Metric Stream]
    end

    subgraph Analytics Service
        TS[Timeline Aggregation (%H:%M:%S)]
        SC[Stability & Consistency Index]
        FL[Focus / Distraction Detector]
        RG[Non-Evaluative Reflective Guidance]
    end

    subgraph Privacy & Ethical Guardrails
        P1[Strictly Private Access]
        P2[No Peer Ranking / Comparisons]
        P3[Zero Grade Penalties]
    end

    PR --> Analytics Service
    PM --> Analytics Service
    Analytics Service --> Dashboard[Student Reflective Dashboard]
    Analytics Service --> Reports[JSON / CSV Session Export]
    Analytics Service --> Privacy
```

---

## 2. Core Architectural Components

### 2.1 Database Schema & Data Models
The analytics engine leverages relational models in SQLite via SQLAlchemy (`backend/core/database.py`):

1. **`SessionPerson` (`session_persons` table)**:
   - Tracks session-level presence aggregates per participant:
     - `session_id` (Integer, Foreign Key)
     - `person_id` (String): Participant identifier (e.g., `STU_1001`)
     - `name` (String, Optional)
     - `presence_duration` (Float): Total seconds participant remained visible
     - `attendance_status` (String): `'PRESENT'` (cumulative $\ge$ threshold) or `'ABSENT'`
     - `average_attention` (Float): Mean attention index over the session

2. **`PersonMetric` (`person_metrics` table)**:
   - High-resolution temporal stream sampled at 1.0s intervals:
     - `session_id` (Integer, Foreign Key)
     - `person_id` (String)
     - `timestamp` (DateTime)
     - `attention` (Float, 0.0 - 100.0%)
     - `emotion` (String: `'engaged'`, `'neutral'`, `'distracted'`, `'confused'`, etc.)

---

### 2.2 Reflective Analytics Pipeline (`AnalyticsService`)
Implemented in `backend/core/analytics_service.py`:

#### 1. Presence Duration Calculation
- Measures active visual presence in seconds and human-readable minutes:
$$\text{Presence Minutes} = \frac{\text{presence\_duration\_sec}}{60.0}$$
- Formatted as `MM:SS` (e.g., `45m 20s`).

#### 2. Engagement Consistency & Stability Index
- Evaluates temporal standard deviation ($\sigma_{\text{att}}$) across individual attention samples:
$$\sigma_{\text{att}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (A_i - \bar{A})^2}$$
- Categorized into reflective stability profiles:
  - $\sigma_{\text{att}} \le 10.0\%$: `HIGHLY_CONSISTENT` (steady, balanced attentiveness)
  - $10.0\% < \sigma_{\text{att}} \le 20.0\%$: `MODERATE_VARIATION` (normal classroom attention ebb and flow)
  - $\sigma_{\text{att}} > 20.0\%$: `FREQUENT_SWINGS` (intermittent high focus punctuated by distraction)

#### 3. Focus & Distraction Timeline Detection
- Scans contiguous temporal buckets to highlight:
  - **High Focus Periods**: Contiguous intervals where attention $\ge 75\%$.
  - **Attention Drop Periods**: Intervals where attention $< 40\%$.
- Timeline segments are logged as human-readable time windows (e.g., `10:15:30 - 10:22:45`).

#### 4. Constructive, Non-Evaluative Guidance
- Provides personalized metacognitive advice without judgment:
  - *Sustained Focus*: Encourages note-taking techniques during peak lecture phases.
  - *Distraction Windows*: Prompts self-awareness (e.g., *"Noticed 2 brief attention drop(s). Taking quick bullet notes or stretching during mid-session transitions can help maintain alertness."*).
  - *Privacy Guarantee*: Explicit notice that data is strictly for personal self-improvement and is never shared with peers or used for grading.

---

## 3. Session Trends Aggregation (`get_session_trends`)
Matching the UI dashboard query requirements:
```python
metrics = db.query(PersonMetric)\
    .filter(PersonMetric.session_id == session_id)\
    .order_by(PersonMetric.timestamp.asc())\
    .all()
```
- Bins high-frequency metric records by second (`%H:%M:%S`).
- Calculates mean attention across all active participants in each bucket.
- Extracts the dominant emotion for the classroom environment.
- Returns downsampled timeline data points for chart rendering (Recharts / Chart.js).

---

## 4. API Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/analytics/session-trends/{session_id}` | Retrieves aggregate temporal attention and emotion trends for the session. |
| `GET` | `/analytics/student-reflective/{session_id}/{student_id}` | Retrieves personalized, non-evaluative participation summary for an individual student. |
| `POST` | `/analytics/record-metric` | Ingests a new temporal metric data point (`session_id`, `person_id`, `attention`, `emotion`). |

---

## 5. Verification & Test Results
Verified via automated test suite `backend/test_student_analytics.py`:
- **Test 1**: Session trends aggregation across time buckets: **PASSED** (mean attention and emotion aggregation verified).
- **Test 2**: Student reflective summary & self-awareness analytics: **PASSED** (high focus periods, attention drop intervals, stability index, and privacy notice confirmed).
- **Test 3**: Sparse student data fallback: **PASSED** (graceful handling of new or missing records).
- **Test 4**: FastAPI `/analytics/*` REST routes: **PASSED** (end-to-end payload handling verified).

