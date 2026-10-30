# Milestone 1: Requirements Definition, Library Selection & Environment Setup
**Project:** Multimodal Attendance & Engagement Agent  
**Repository:** `Shresthaa011/MultimodalAttendance-EngagementAgent`  
**Milestone:** 1 - System Specifications, Architecture & Foundation  

---

## 1. Executive Summary & Objective

The primary objective of this milestone is to establish a well-defined, reproducible, and scalable foundation for the **Multimodal Attendance & Engagement Agent**. This agent is designed to automate attendance logging and quantify real-time classroom/meeting engagement by fusing visual cues (facial presence, head pose, eye gaze, blink rates) and acoustic signals (voice activity, speech presence, ambient noise levels), augmented by an LLM-powered reasoning agent (Groq) for automated insights and report generation.

Milestone 1 accomplishes:
1. Complete formalization of **Functional Requirements (FR)** and **Non-Functional Requirements (NFR)**.
2. Comprehensive selection and categorization of third-party libraries across Vision, Audio, Backend/API, and Data/AI layers.
3. Verification of dependency compatibility with target Python environments.
4. Structuring of the development repository with reproducibility tools (`requirements.txt`, `.env.example`, `.gitignore`, `verify_env.py`, and `test_pipeline.py`).
5. Provisioning of an isolated virtual environment and validation through live server invocation.

---

## 2. Approach & Engineering Methodology

Developing a real-time multimodal system requires balancing computational throughput (FPS, low-latency audio processing) with analytical precision. The approach followed these sequential stages:

1. **Functional Decomposition**: Dissected user workflows (e.g., student check-in, continuous presence tracking, live engagement scoring, anomaly alert, session summary).
2. **Multimodal Dependency Mapping**: Mapped each capability to battle-tested, open-source libraries prioritized for execution speed, documentation depth, and active community maintenance.
3. **Cross-Platform Compatibility Audit**: Evaluated Python package availability (pre-built wheels, C/C++ backend bindings for OpenCV and SoundFile) on Windows and Linux targets.
4. **Environment Isolation**: Designed a dedicated virtual environment configuration ensuring version determinism and conflict-free coexistence across media pipelines.

---

## 3. Process Execution

### 3.1 Feature Analysis & Functional Decomposition
The system was analyzed across five core functional modules:
- **Vision Processing**: Face detection, facial landmark localization, head orientation (yaw, pitch, roll), eye blink/aspect ratio (EAR), and persistent multi-subject tracking across frame occlusions.
- **Audio Analysis**: Acoustic signal ingestion, Voice Activity Detection (VAD), speech segment extraction, signal-to-noise ratio (SNR) calculation, and acoustic energy levels.
- **Multimodal Fusion**: Cross-modality temporal alignment combining visual attention scores and acoustic participation into a normalized composite Engagement Index ($0.0 - 1.0$).
- **Backend & Streaming API**: Low-latency bi-directional WebSocket streaming for telemetry/video frames, REST endpoints for management, and transactional database storage.
- **Agentic Insights & Analytics**: LLM-driven anomaly detection (e.g., flagging disengaged clusters, prolonged absences, or high ambient distractions) and automated report generation.

### 3.2 Library Identification & Module Mapping
Based on feature requirements, the following software stack was selected:

| Functional Module | Selected Libraries | Key Capabilities |
| :--- | :--- | :--- |
| **Vision Processing** | `opencv-python`<br>`mediapipe`<br>`deep-sort-realtime` | Frame decoding, 468-point 3D Face Mesh, iris landmarking, Kalman filter-based multi-object tracking. |
| **Audio Analysis** | `librosa`<br>`soundfile` | Audio I/O, spectrogram analysis, RMS energy computation, zero-crossing rates, acoustic feature extraction. |
| **Backend & API Handling** | `fastapi`<br>`uvicorn`<br>`websockets`<br>`python-multipart`<br>`sqlalchemy`<br>`python-dotenv` | ASGI async architecture, WebSocket streaming, file upload handling, ORM data persistence, environment isolation. |
| **Data Storage & Analytics** | `numpy`<br>`pandas`<br>`groq` | Vectorized mathematical operations, time-series data aggregation, automated CSV/Excel reporting, ultra-fast LLM inference. |

### 3.3 Compatibility Verification & Elimination of Conflicts
- Confirmed that `opencv-python` and `mediapipe` share compatible NumPy array representations without requiring expensive frame conversions.
- Validated that `fastapi` and `websockets` integrate seamlessly for concurrent real-time audio/video streaming.
- Verified that `soundfile` (via `libsndfile`) provides low-overhead buffer reading suitable for chunked audio streams without disk writes.
- Eliminated redundant computer vision libraries (e.g., avoiding heavy single-purpose face detectors when MediaPipe provides both mesh and bounding detection).

---

## 4. Requirements Specification

### 4.1 Functional Requirements (FR)

| ID | Module | Description |
| :--- | :--- | :--- |
| **FR-01** | Attendance | The system shall automatically detect and recognize registered individuals within the camera field-of-view upon entry. |
| **FR-02** | Tracking | The system shall assign persistent track IDs to detected subjects using DeepSORT to avoid duplicate check-ins during frame dropouts or temporary occlusions. |
| **FR-03** | Visual Attention | The system shall compute Head Pose (Pitch, Yaw, Roll) and Eye Aspect Ratio (EAR) from 468 MediaPipe facial landmarks to classify head-up attentive vs. head-down/distracted states. |
| **FR-04** | Audio Cues | The system shall detect active speech segments and quantify ambient noise levels using Librosa and SoundFile chunked signal analysis. |
| **FR-05** | Fusion Engine | The system shall fuse visual gaze vectors and audio activity over configurable sliding windows (e.g., 5-second intervals) to generate normalized engagement scores ($0.0 - 1.0$). |
| **FR-06** | Agentic Reporting | The system shall utilize Groq-accelerated LLM reasoning to generate structured session summaries, highlight engagement dips, and produce downloadable reports (CSV/JSON/PDF). |
| **FR-07** | Streaming API | The system shall provide WebSocket endpoints streaming real-time bounding boxes, landmark overlays, and engagement telemetry to connected client dashboards. |
| **FR-08** | Data Persistence | The system shall store student profiles, attendance timestamps, session metadata, and engagement time-series records in a relational database via SQLAlchemy. |

### 4.2 Non-Functional Requirements (NFR)

| ID | Category | Requirement Details |
| :--- | :--- | :--- |
| **NFR-01** | **Performance & Throughput** | The vision pipeline shall process live video streams at a minimum of $15-20$ FPS on standard CPU hardware, with latency below $150$ ms per frame. |
| **NFR-02** | **Audio Latency** | Audio feature extraction shall execute on overlapping $1.0 - 2.0$ second window chunks with processing latency $< 100$ ms. |
| **NFR-03** | **Scalability & Concurrency** | The backend shall leverage FastAPI's asynchronous event loop (`asyncio`) to handle concurrent client sessions without blocking computational worker threads. |
| **NFR-04** | **Data Privacy & Ethics** | Biometric feature vectors and video frames shall be processed in-memory. Only anonymous feature metrics and attendance logs shall be persisted, respecting user privacy. |
| **NFR-05** | **Reproducibility** | The development environment shall be 100% reproducible through version-controlled dependencies (`requirements.txt`), automated verification scripts (`verify_env.py`), and configuration templates (`.env.example`). |
| **NFR-06** | **Reliability & Resilience** | The system shall gracefully handle webcam or microphone disconnection, resuming telemetry without crashing the host server. |

---

## 5. Technical Details & Dependency Architecture

The backend dependencies in `backend/requirements.txt` are categorized as follows:

```
# Computer Vision
opencv-python
mediapipe
deep-sort-realtime

# Audio Processing
librosa
soundfile

# Backend & API
fastapi
uvicorn
websockets
python-multipart
sqlalchemy
python-dotenv

# Data Processing & LLM
numpy
pandas
groq
```

### Detailed Rationale per Component:

1. **Computer Vision Layer**:
   - `opencv-python`: Provides fundamental image transformations, color space conversions ($BGR \to RGB$), drawing utilities for video overlays, and webcam acquisition.
   - `mediapipe`: Google's lightweight ML solution delivering 468 3D facial landmarks with sub-millisecond execution, enabling real-time head pose estimation and blink detection without heavy GPU dependencies.
   - `deep-sort-realtime`: Integrates Kalman filtering and appearance embeddings for reliable multi-person tracking across complex classroom or conference environments.

2. **Audio Processing Layer**:
   - `librosa`: Industry standard for audio feature extraction (RMS energy, spectral centroid, MFCCs) used to quantify vocal activity and distinguish speech from background classroom noise.
   - `soundfile`: Provides C-level bindings (`libsndfile`) for ultra-low latency reading/writing of raw PCM audio buffers streaming from client microphones.

3. **Backend & API Layer**:
   - `fastapi`: Modern, high-performance web framework featuring automatic OpenAPI documentation, Pydantic type safety, and native asynchronous request handling.
   - `uvicorn`: ASGI server providing blazing fast event loop execution via `uvloop`.
   - `websockets`: Powers full-duplex communication channels between the backend multimodal engine and client monitoring interfaces.
   - `python-multipart`: Handles streaming multi-part HTTP requests for student roster uploads and recorded media files.
   - `sqlalchemy`: Flexible SQL toolkit and ORM ensuring portable, structured schema management across SQLite (development) and PostgreSQL (production).
   - `python-dotenv`: Enforces 12-Factor App methodology by loading API keys, model parameters, and database URLs from environment files.

4. **Data Processing & LLM Agent Layer**:
   - `numpy`: High-speed vector math for coordinate normalization, distance calculations (EAR), and matrix operations.
   - `pandas`: Structured data manipulation for aggregating second-by-second engagement metrics into session-level attendance reports.
   - `groq`: Ultra-fast inference engine for LPU-hosted open-weights models (e.g., LLaMA-3.3-70B), performing natural-language session diagnostics, anomaly identification, and instructor recommendations in near real-time.

---

## 6. Environment Setup, Virtual Environment Isolation & Verification

### 6.1 Approach
To ensure complete isolation from system-level Python packages and prevent dependency version conflicts, a dedicated virtual environment was created for the project. This approach improves reproducibility, isolates heavy C/C++ wheel binaries (such as OpenCV and SoundFile), and aligns strictly with best practices for professional software engineering.

### 6.2 Process
1. **Verification of Python Installation:** Confirmed Python 3.13 (64-bit AMD64) and pip package manager runtime availability.
2. **Virtual Environment Creation:** Created an isolated virtual environment (`backend/venv`) dedicated specifically to the Multimodal Agent backend.
3. **Activation:** Activated the virtual environment shell prior to package operations to ensure local isolation.
4. **Automated Package Installation:** Installed all 14 required libraries across Computer Vision, Audio, Backend/API, and Data/LLM tiers via a single command referencing `requirements.txt`.
5. **System Validation & Smoke Testing:**
   - Executed `verify_env.py` to assert that all 14 libraries can be imported and report resolved version numbers.
   - Executed `test_pipeline.py` to validate synthetic OpenCV image drawing, Librosa RMS audio feature calculation, and FastAPI app configuration.
   - Launched the backend server using Uvicorn and verified `/health` and `/` endpoints via live HTTP requests.

### 6.3 Technical Details
- **Environment Tooling:** Standard library `venv` module (`python -m venv venv`) paired with `pip`.
- **Package Manifest:** `backend/requirements.txt` grouping 14 critical dependencies.
- **Verification Utility:** Custom `verify_env.py` dynamically importing module bindings and displaying formatted status matrices.
- **Smoke Test Suite:** `test_pipeline.py` exercising image matrix synthesis, synthetic audio signal processing, and ASGI application routing.
- **Server Execution:** High-performance ASGI runtime (`uvicorn app.main:app`) reading configuration variables from `.env`.

### 6.4 Verification & Diagnostic Execution Logs

#### 1. Automated Environment Diagnostic (`verify_env.py`)
```text
======================================================================
 MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT - ENVIRONMENT VERIFIER
======================================================================
Python Version: 3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]
Platform: win32
======================================================================

[Category: Computer Vision]
--------------------------------------------------
  [ OK ] opencv-python        (import: cv2               ) v5.0.0
  [ OK ] mediapipe            (import: mediapipe         ) v1.2.0
  [ OK ] deep-sort-realtime   (import: deep_sort_realtime) v1.3.2

[Category: Audio Processing]
--------------------------------------------------
  [ OK ] librosa              (import: librosa           ) v1.0.0
  [ OK ] soundfile            (import: soundfile         ) v0.14.0

[Category: Backend & API]
--------------------------------------------------
  [ OK ] fastapi              (import: fastapi           ) v0.162.0
  [ OK ] uvicorn              (import: uvicorn           ) v0.54.0
  [ OK ] websockets           (import: websockets        ) v17.3
  [ OK ] python-multipart     (import: multipart         ) v0.0.32
  [ OK ] sqlalchemy           (import: sqlalchemy        ) v2.1.6
  [ OK ] python-dotenv        (import: dotenv            ) vinstalled (no __version__)

[Category: Data Processing & LLM]
--------------------------------------------------
  [ OK ] numpy                (import: numpy             ) v2.5.4
  [ OK ] pandas               (import: pandas            ) v3.1.0
  [ OK ] groq                 (import: groq              ) v1.7.0

======================================================================
Summary: 14/14 packages verified.
All dependencies are successfully installed and verified!
======================================================================
```

#### 2. Pipeline Integration & Smoke Test (`test_pipeline.py`)
```text
============================================================
 MULTIMODAL AGENT - INTEGRATION & SMOKE TEST
============================================================
Testing module imports...
All modules imported successfully!
Testing basic OpenCV & NumPy operation...
OpenCV frame creation & draw operation: PASSED
Testing basic audio processing with NumPy & Librosa...
Librosa audio feature extraction (RMS shape=(1, 32)): PASSED
Testing FastAPI app initialization...
FastAPI application configuration: PASSED
============================================================
ALL TESTS PASSED SUCCESSFULLY! ENVIRONMENT IS READY.
============================================================
```

#### 3. Backend Server Health Check
```text
INFO:     Started server process [11440]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:49922 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:49926 - "GET / HTTP/1.1" 200 OK

Response Body:
{
    "status": "healthy",
    "service": "Multimodal Attendance & Engagement Agent",
    "version": "1.0.0"
}
```

---

## 7. Milestone Conclusion & Next Steps

Milestone 1 successfully establishes:
- The architectural blueprint and requirements baseline for multimodal tracking and engagement intelligence.
- A fully defined dependency manifest (`requirements.txt`) structured into modular subsystems.
- An isolated, verified virtual environment with 100% dependency resolution.
- Live server invocation confirming working REST and health-check routing.

**Upcoming Objectives (Milestone 2):**
1. Implementation of the Computer Vision Pipeline (`MediaPipe` Face Mesh + Head Pose calculation + `DeepSORT` tracker).
2. Development of the Audio Processing Module (VAD + Ambient Noise estimation).
3. Establishing the WebSocket streaming ingest server in `FastAPI`.
