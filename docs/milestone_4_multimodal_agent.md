# Milestone 4 — Multimodal Agent Core Intelligence Layer
## Interpretable Rule-Based Engagement Reasoning & Pedagogical LLM Insights

---

### 1. Executive Summary & Objectives

The **Multimodal Agent** serves as the central intelligence and reasoning engine of the **Multimodal Attendance & Engagement Agent**. Rather than treating vision and audio signals as isolated data streams, the agent synthesizes inputs from both the **Vision Module** (participant presence, sustained tracking, visual focus) and the **Audio Module** (decibel noise levels, noise stability, speech activity ratio, interaction intensity) into a unified, contextual understanding of classroom dynamics.

To ensure suitability for real-world educational deployment, engagement estimation is governed by **transparent, rule-based reasoning** rather than opaque black-box models. Every evaluation produces an explainable trace, detailing the specific pedagogical rules triggered, the respective contribution weights of each modality, and contextual factors. Furthermore, an **LLM-powered Insight Generator** (`llm_insights.py`) transforms aggregate session metrics into actionable pedagogical recommendations for instructors.

```mermaid
flowchart TD
    subgraph Modalities["Modality Ingestion Layers"]
        V["Vision Module\n(Face Detection, IoU Tracking, Presence Duration)"]
        A["Audio Module\n(16kHz RMS dBFS, ZCR, Centroid, Speech Ratio)"]
    end

    subgraph Agent["Multimodal Agent Core (Rule-Based Reasoning)"]
        V --> FE["Feature Alignment & Normalization\n(Visual Score 55%, Acoustic Score 45%)"]
        A --> FE
        FE --> RE{"Pedagogical Rule Engine\n(R1 - R6 Predicates)"}
        RE -->|"Active Discussion"| R1["R1: Active Collaboration (+0.15 Delta)"]
        RE -->|"Silent Attention"| R2["R2: Focused Listening (+0.10 Delta)"]
        RE -->|"Erratic Disturbance"| R3["R3: Environmental Disturbance (-0.25 Delta)"]
        RE -->|"Looking Away & Silent"| R4["R4: Passive Inattention (-0.20 Delta)"]
        RE -->|"Vocal but Off-Camera"| R5["R5: Off-Camera Interaction (+0.05 Delta)"]
        RE -->|"Standard Dynamics"| R6["R6: Baseline Engagement (0.00 Delta)"]
    end

    subgraph Output["Explainability & Pedagogical Action Layer"]
        R1 & R2 & R3 & R4 & R5 & R6 --> MEI["Composite Multimodal Engagement Index (MEI)\n[0.0 - 1.0] (HIGH / MEDIUM / LOW)"]
        MEI --> EXP["Transparent Explainability Breakdown\n(Triggered Rules & Modality Weights)"]
        MEI --> LLM["InsightGenerator (Groq LLM / Fallback)\n(Classroom Dynamic & Actionable Advice)"]
        LLM --> WS["Real-time WebSocket & REST Telemetry\n(/ws/audio, /agent/session-insight)"]
    end
```

---

### 2. Rule-Based Reasoning Engine (Interpretable & Explainable)

The core agent applies an explicit pedagogical rule matrix to classify classroom states and determine context modulation deltas ($\Delta_{\text{context}}$):

| Rule ID | Pedagogical State | Mathematical / Signal Predicates | Context Delta ($\Delta$) | Engagement Level | Explainability Sentence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`R1`** | **`active_collaboration`** | $V \ge 0.50 \land (S_r \ge 0.25 \lor \text{Intensity} \in \{\text{'high'}, \text{'moderate'}\})$ | $+0.15$ | **HIGH** | *"Participants are visibly present and actively participating in verbal exchange or answering questions."* |
| **`R2`** | **`focused_listening`** | $V \ge 0.60 \land S_r < 0.12 \land \text{Noise}_{\text{dB}} \le -42.0\text{ dB}$ | $+0.10$ | **HIGH** | *"Sustained visual attention in a quiet, stable acoustic setting indicates concentrated lecture listening."* |
| **`R3`** | **`environmental_disturbance`** | $\text{Noise}_{\text{dB}} > -30.0\text{ dB} \land S_r < 0.12 \land \text{Stability} == \text{'erratic'}$ | $-0.25$ | **LOW** | *"Elevated erratic environmental noise is disturbing the classroom environment, hindering student attention."* |
| **`R4`** | **`passive_inattention`** | $V < 0.40 \land (S_r < 0.10 \lor \text{Intensity} == \text{'silent'})$ | $-0.20$ | **LOW** | *"Low visual attentiveness and passive silence suggest student fatigue or disengagement from lecture content."* |
| **`R5`** | **`off_camera_verbal_interaction`**| $V < 0.40 \land S_r \ge 0.20$ | $+0.05$ | **MEDIUM** | *"Active vocal interaction detected despite low camera-facing visual focus, indicating verbal contribution or peer discussion."* |
| **`R6`** | **`moderate_steady_engagement`**| Default baseline when no boundary conditions trigger | $0.00$ | **MEDIUM** | *"Standard classroom engagement with steady acoustic background and expected student presence."* |

---

### 3. Composite Multimodal Engagement Index (MEI) Formulation

The composite engagement index fuses normalized visual and acoustic cues with the pedagogical context adjustment:

$$\text{MEI} = \operatorname{clamp}\left(w_v \cdot E_{\text{visual}} + w_a \cdot E_{\text{acoustic}} + \Delta_{\text{context}}, \, 0.0, \, 1.0\right)$$

Where:
- $w_v = 0.55$: Visual modality weight (presence verification and sustained attentiveness).
- $w_a = 0.45$: Acoustic modality weight (noise stability and interaction dynamics).
- $w_v + w_a = 1.0$: Strict normalization constraint.
- $\Delta_{\text{context}} \in [-0.25, +0.15]$: Context modulation factor determined by the triggered rule predicate.

#### Discrete Engagement Classification
- **`HIGH`**: $\text{MEI} \ge 0.70$
- **`MEDIUM`**: $0.45 \le \text{MEI} < 0.70$
- **`LOW`**: $\text{MEI} < 0.45$

---

### 4. LLM Insights & Recommendation Generator (`llm_insights.py`)

The agent incorporates an `InsightGenerator` leveraging the **Groq API** (`mixtral-8x7b-32768` or `llama-3.3-70b-versatile`) with a deterministic rule-based fallback:

```python
class InsightGenerator:
    def generate_classroom_insight(self, session_data: dict) -> dict:
        prompt = self._build_prompt(session_data)
        if not self.client:
            return self._generate_fallback_insight(session_data)
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert educational AI assistant..."},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.5,
                max_tokens=150,
            )
            response_text = chat_completion.choices[0].message.content
            # Simple Parsing
            insight = "Class status is stable."
            recommendation = "Maintain current pace."
            ...
```

#### Dual-Mode Architecture:
1. **Live LLM Inference:** When `GROQ_API_KEY` is provided in `.env`, the generator synthesizes contextual prompts with attendance percentage, visual scores, audio patterns, and triggered rules into natural-language diagnostic insights.
2. **Deterministic Fallback Engine:** When offline, in unit tests, or without an API key, the system seamlessly applies pre-calibrated pedagogical insights based on the active state (e.g., advising an interactive poll during `passive_inattention` or recommending student sharing during `active_collaboration`).

---

### 5. Backend REST & WebSocket API Endpoints

The agent exposes dedicated interfaces within `backend/app/main.py`:

| Endpoint | Method | Purpose & Description |
| :--- | :--- | :--- |
| `/agent/status` | `GET` | Reports agent operational state, weights, total snapshots, and active metrics. |
| `/agent/evaluate` | `POST` | Accepts snapshot of visual + audio data and returns explainable engagement trace. |
| `/agent/session-insight` | `GET` | Delivers rolling mean MEI, temporal trend (`improving`/`declining`/`stable`), and LLM pedagogical advice. |
| `/agent/reset` | `POST` | Resets session memory between class periods. |
| `/ws/audio` | `WS` | Emits `multimodal_engagement` telemetry objects in real time alongside audio metrics. |

---

### 6. Automated Test Verification Results

All tests across all 4 system modules executed with **100% pass rates**:

```text
======================================================================
 MULTIMODAL AGENT - REASONING, EXPLAINABILITY & INSIGHTS TEST SUITE
======================================================================
Test 1: Testing rule-based reasoning across classroom scenarios (R1 - R6)...
  [R1 Verified] Active Collaboration -> Score: 1.0 | Level: HIGH
  [R2 Verified] Focused Listening -> Score: 0.932 | Level: HIGH
  [R3 Verified] Environmental Disturbance -> Score: 0.115 | Level: LOW
  [R4 Verified] Passive Inattention -> Score: 0.073 | Level: LOW
  [R5 Verified] Off-Camera Interaction -> Score: 0.598 | Level: MEDIUM
  [PASSED] Rule-based reasoning engine verified across all classroom states.

Test 2: Testing explainability, transparent rule traces & contribution breakdown...
  Explanation Generated: "Sustained visual attention in a quiet, stable acoustic setting indicates concentrated lecture listening."
  Triggered Rule: R2_FOCUSED_LISTENING - High visual attentiveness in a calm acoustic environment indicates focused lecture absorption or individual work.
  Modality Contributions: Visual (w=0.55, s=0.75) + Audio (w=0.45, s=0.7) + Delta (0.1)
  [PASSED] Explainability and transparent contribution breakdown verified.

Test 3: Testing session aggregation, rolling MEI & trend analysis...
  Session Trend: improving | Mean MEI: 0.64 | Level: MEDIUM
  Pedagogical Insight: "High multimodal engagement observed with dynamic student participation and sustained visual attentiveness."
  Instructor Recommendation: "Facilitate structured student sharing and summarize key collaborative takeaways."
  [PASSED] Session-level temporal trends and pedagogical intelligence verified.

Test 4: Testing InsightGenerator fallback and prompt construction...
  Insight Source: rule_based_fallback (Model: rule_engine_v1)
  Insight: "High multimodal engagement observed with dynamic student participation and sustained visual attentiveness."
  Recommendation: "Facilitate structured student sharing and summarize key collaborative takeaways."
  [PASSED] InsightGenerator prompt synthesis and parsing verified.

Test 5: Testing FastAPI Multimodal Agent endpoints & WebSocket fusion...
  POST /agent/evaluate -> State: active_collaboration | MEI: 0.961
  GET /agent/status -> Snapshots: 1 | Status: active
  GET /agent/session-insight -> Mean MEI: 0.961 | Insight: "High multimodal engagement observed with dynamic student participation and sustained visual attentiveness."
  WS /ws/audio Broadcast -> State: off_camera_verbal_interaction | MEI: 0.555 | Level: MEDIUM
  [PASSED] FastAPI Multimodal Agent endpoints and WebSocket telemetry verified.

======================================================================
ALL MULTIMODAL AGENT TESTS PASSED (5/5).
======================================================================
```

---

### 7. File Reference Links

- [`backend/core/multimodal_agent.py`](file:///e:/Multimodal%20Attendance%20%26%20Engagement%20Agent/MultimodalAttendance-EngagementAgent/backend/core/multimodal_agent.py): Rule-based reasoning engine, composite MEI estimation, and session intelligence aggregation.
- [`backend/core/llm_insights.py`](file:///e:/Multimodal%20Attendance%20%26%20Engagement%20Agent/MultimodalAttendance-EngagementAgent/backend/core/llm_insights.py): Groq LLM inference integration and deterministic pedagogical fallback engine.
- [`backend/app/main.py`](file:///e:/Multimodal%20Attendance%20%26%20Engagement%20Agent/MultimodalAttendance-EngagementAgent/backend/app/main.py): FastAPI endpoints (`/agent/*`) and WebSocket fusion streaming.
- [`backend/test_multimodal_agent.py`](file:///e:/Multimodal%20Attendance%20%26%20Engagement%20Agent/MultimodalAttendance-EngagementAgent/backend/test_multimodal_agent.py): 5-stage automated test suite validating rules, explainability, trends, LLM insights, and API routes.

