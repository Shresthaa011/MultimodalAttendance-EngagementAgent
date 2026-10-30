"""
Audio Processing & Multi-Source Ingestion Pipeline
Multimodal Attendance & Engagement Agent

Handles:
1. Audio extraction from recorded video/audio files or raw microphone streams.
2. Sampling rate normalization (default 16 kHz) and channel standardization (mono).
3. Fixed-duration window segmentation aligned with video timelines.
4. Time-indexed structured storage (JSON & CSV) for multimodal fusion.
5. Privacy-preserving analysis (zero speech-to-text / acoustic content inspection).
"""

import os
import json
import time
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from .audio_analysis import AudioAnalyzer

class AudioProcessor:
    """
    Processes environmental audio from files or streaming buffers,
    normalizes signals, segments into synchronized windows, and exports structured logs.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        window_duration_sec: float = 1.0,
        hop_duration_sec: float = 1.0,
        log_dir: str = "logs/audio",
    ):
        self.sample_rate = sample_rate
        self.window_duration_sec = window_duration_sec
        self.hop_duration_sec = hop_duration_sec
        self.log_dir = log_dir
        self.analyzer = AudioAnalyzer(sample_rate=sample_rate)
        self.window_samples = int(self.sample_rate * self.window_duration_sec)
        self.hop_samples = int(self.sample_rate * self.hop_duration_sec)

        # Buffer for continuous streaming input
        self.stream_buffer = np.array([], dtype=np.float32)
        self.processed_windows: List[Dict[str, Any]] = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def normalize_waveform(self, audio_data: Union[np.ndarray, bytes], original_sr: Optional[int] = None) -> np.ndarray:
        """
        Converts raw audio bytes or arrays into normalized, mono, float32 digital waveforms.
        Applies resample normalization to target sample rate (default: 16 kHz).
        """
        if isinstance(audio_data, bytes):
            y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            y = np.asarray(audio_data, dtype=np.float32)

        # Multi-channel standardization: convert stereo/multi-channel to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Sampling rate normalization
        if original_sr is not None and original_sr != self.sample_rate and len(y) > 0:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=self.sample_rate)

        return y

    def load_and_preprocess_file(self, file_path: str) -> np.ndarray:
        """
        Loads audio from an audio or video file (.mp4, .wav, .mp3, .m4a),
        standardizing to mono and 16 kHz.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio/Video source file not found: {file_path}")

        # Librosa automatically handles mono conversion and sampling rate normalization
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return y.astype(np.float32)

    def segment_waveform(self, y: np.ndarray, base_timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """
        Segments continuous waveform into fixed-duration windows,
        calculating timestamps and acoustic engagement metrics for each window.
        """
        total_samples = len(y)
        results = []
        window_idx = 0

        for start_sample in range(0, total_samples, self.hop_samples):
            end_sample = start_sample + self.window_samples
            chunk = y[start_sample:end_sample]

            # If the last chunk is shorter than window_samples, pad with zeros
            if len(chunk) < self.window_samples:
                if len(chunk) < self.window_samples // 4:
                    # Skip negligible trailing slivers
                    break
                chunk = np.pad(chunk, (0, self.window_samples - len(chunk)), mode='constant')

            start_sec = round(start_sample / self.sample_rate, 3)
            end_sec = round((start_sample + len(chunk)) / self.sample_rate, 3)
            abs_start = round(base_timestamp + start_sec, 3)
            abs_end = round(base_timestamp + end_sec, 3)

            # Analyze chunk
            analysis = self.analyzer.analyze_audio_chunk(chunk)
            if analysis is None:
                continue

            window_record = {
                "window_index": window_idx,
                "relative_start_sec": start_sec,
                "relative_end_sec": end_sec,
                "absolute_start_timestamp": abs_start,
                "absolute_end_timestamp": abs_end,
                **analysis,
            }
            results.append(window_record)
            window_idx += 1

        self.processed_windows.extend(results)
        return results

    def process_streaming_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        chunk_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ingests real-time audio chunk from microphone or streaming WebSocket,
        buffers until window duration is reached, and yields completed window metrics.
        """
        y_chunk = self.normalize_waveform(audio_chunk)
        self.stream_buffer = np.concatenate([self.stream_buffer, y_chunk])

        current_time = chunk_timestamp or time.time()
        completed_windows = []

        while len(self.stream_buffer) >= self.window_samples:
            window_data = self.stream_buffer[:self.window_samples]
            self.stream_buffer = self.stream_buffer[self.hop_samples:]

            analysis = self.analyzer.analyze_audio_chunk(window_data)
            if analysis:
                window_record = {
                    "window_index": len(self.processed_windows),
                    "timestamp": round(current_time, 3),
                    "duration_sec": self.window_duration_sec,
                    **analysis,
                }
                self.processed_windows.append(window_record)
                completed_windows.append(window_record)

        return completed_windows

    def save_structured_logs(self, session_id: Optional[str] = None) -> Dict[str, str]:
        """
        Exports accumulated audio analysis windows to structured JSON and CSV files.
        """
        sid = session_id or f"audio_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not self.processed_windows:
            avg_noise_db = -60.0
            overall_speech_ratio = 0.0
            dominant_pattern = "silence"
        else:
            avg_noise_db = round(float(np.mean([w["noise_level_db"] for w in self.processed_windows])), 2)
            overall_speech_ratio = round(float(np.mean([w["speech_ratio"] for w in self.processed_windows])), 3)
            # Find dominant activity pattern
            patterns = [w["activity_pattern"] for w in self.processed_windows]
            dominant_pattern = max(set(patterns), key=patterns.count)

        payload = {
            "session_id": sid,
            "sample_rate_hz": self.sample_rate,
            "window_duration_sec": self.window_duration_sec,
            "total_windows_processed": len(self.processed_windows),
            "average_noise_level_db": avg_noise_db,
            "overall_speech_ratio": overall_speech_ratio,
            "dominant_activity_pattern": dominant_pattern,
            "exported_at": datetime.now().isoformat(),
            "windows": self.processed_windows,
        }

        # 1. JSON Export
        json_path = os.path.join(self.log_dir, f"{sid}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # 2. CSV Export
        csv_path = os.path.join(self.log_dir, f"{sid}.csv")
        if self.processed_windows:
            df = pd.DataFrame(self.processed_windows)
            df.to_csv(csv_path, index=False)
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("window_index,relative_start_sec,relative_end_sec,noise_level_db,speech_detected,speech_ratio,activity_pattern,acoustic_engagement_score\n")

        return {
            "json_log_path": json_path,
            "csv_report_path": csv_path,
        }

    def get_session_audio_context(self, recent_window_count: int = 10) -> Dict[str, Any]:
        """
        Aggregates recent and session-level acoustic signals into a structured format
        optimized for multimodal fusion with visual engagement data.
        """
        total_windows = len(self.processed_windows)
        if total_windows == 0:
            return {
                "timestamp": round(time.time(), 3),
                "windows_analyzed": 0,
                "noise_metrics": {
                    "current_noise_level_db": -60.0,
                    "mean_noise_level_db": -60.0,
                    "noise_stability": "calm",
                },
                "speech_dynamics": {
                    "speech_detected": False,
                    "recent_speech_ratio": 0.0,
                    "interaction_intensity": "silent",
                    "speech_bursts_count": 0,
                },
                "multimodal_fusion_cues": {
                    "acoustic_engagement_score": 0.50,
                    "classroom_activity_state": "quiet_listening",
                },
            }

        recent_windows = self.processed_windows[-recent_window_count:]
        recent_noise = [w["noise_level_db"] for w in recent_windows]
        recent_speech = [w["speech_ratio"] for w in recent_windows]
        recent_scores = [w["acoustic_engagement_score"] for w in recent_windows]

        mean_db = round(float(np.mean(recent_noise)), 2)
        std_db = round(float(np.std(recent_noise)), 2) if len(recent_noise) > 1 else 0.0
        current_db = recent_noise[-1]

        # Determine noise stability
        if std_db < 3.0:
            noise_stability = "steady"
        elif std_db < 7.0:
            noise_stability = "moderate_fluctuation"
        else:
            noise_stability = "erratic_disturbance"

        # Determine speech interaction intensity
        mean_speech = round(float(np.mean(recent_speech)), 3)
        speech_detected_recently = any(w["speech_detected"] for w in recent_windows)

        if mean_speech >= 0.40:
            interaction_intensity = "high_interaction"
        elif mean_speech >= 0.15:
            interaction_intensity = "moderate_participation"
        elif speech_detected_recently:
            interaction_intensity = "low_intermittent"
        else:
            interaction_intensity = "silent_listening"

        # Count speech turns / transitions across the session
        speech_turns = 0
        in_speech = False
        for w in self.processed_windows:
            if w["speech_detected"] and not in_speech:
                speech_turns += 1
                in_speech = True
            elif not w["speech_detected"]:
                in_speech = False

        # Dominant activity pattern in recent window
        patterns = [w["activity_pattern"] for w in recent_windows]
        classroom_state = max(set(patterns), key=patterns.count)
        acoustic_score = round(float(np.mean(recent_scores)), 3)

        return {
            "timestamp": round(time.time(), 3),
            "windows_analyzed": total_windows,
            "noise_metrics": {
                "current_noise_level_db": current_db,
                "mean_noise_level_db": mean_db,
                "noise_std_dev": std_db,
                "noise_stability": noise_stability,
            },
            "speech_dynamics": {
                "speech_detected": speech_detected_recently,
                "recent_speech_ratio": mean_speech,
                "interaction_intensity": interaction_intensity,
                "speech_bursts_count": speech_turns,
            },
            "multimodal_fusion_cues": {
                "acoustic_engagement_score": acoustic_score,
                "classroom_activity_state": classroom_state,
            },
        }

