import librosa
import numpy as np
import os
from typing import Dict, Any, Optional, Union

class AudioAnalyzer:
    def __init__(self, sample_rate: int = 16000, speech_threshold_rms: float = 0.015, noise_threshold_db: float = -35.0):
        self.sample_rate = sample_rate
        self.speech_threshold_rms = speech_threshold_rms
        self.noise_threshold_db = noise_threshold_db

    def analyze_audio_chunk(self, audio_data: Union[bytes, np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Analyzes raw audio bytes/array:
        1. Noise level (decibels)
        2. Speech vs silence ratio
        3. Sound activity patterns

        Returns engagement metrics
        """
        try:
            # Convert bytes to float32 numpy array if needed
            if isinstance(audio_data, bytes):
                # Assuming 16-bit PCM
                y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                y = np.asarray(audio_data, dtype=np.float32)

            if len(y) == 0:
                return None

            # Handle multi-channel arrays by converting to mono
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # 1. Noise level and RMS Energy
            # Calculate RMS over sub-frames (frame_length 512, hop_length 256)
            frame_length = min(512, len(y))
            hop_length = max(1, frame_length // 2)

            frame_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            mean_rms = float(np.mean(frame_rms)) if len(frame_rms) > 0 else float(np.sqrt(np.mean(y**2)))

            # Convert to decibels (dBFS)
            # Clip minimum at 1e-5 (-100 dBFS floor)
            noise_level_db = float(20.0 * np.log10(max(mean_rms, 1e-5)))
            noise_level_db = round(max(-100.0, min(0.0, noise_level_db)), 2)

            # 2. Zero Crossing Rate (ZCR) for speech articulation cues
            zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            mean_zcr = float(np.mean(zcr)) if len(zcr) > 0 else 0.0

            # 3. Spectral Centroid (frequency brightness / energy center)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=self.sample_rate, n_fft=frame_length, hop_length=hop_length
            )[0]
            mean_centroid = float(np.mean(spectral_centroid)) if len(spectral_centroid) > 0 else 0.0

            # 4. Speech vs Silence Ratio (Privacy-preserving behavioral classification)
            # Human conversational speech has:
            # - Energy > baseline threshold
            # - ZCR within speech articulatory bounds (0.02 to 0.35)
            # - Spectral centroid typically between 300 Hz and 3800 Hz
            speech_frames = (
                (frame_rms >= self.speech_threshold_rms) &
                (zcr >= 0.015) &
                (zcr <= 0.35) &
                (spectral_centroid >= 250.0) &
                (spectral_centroid <= 4000.0)
            )

            silence_frames = frame_rms < (self.speech_threshold_rms * 0.5)

            total_frames = max(1, len(frame_rms))
            speech_ratio = float(np.sum(speech_frames)) / total_frames
            silence_ratio = float(np.sum(silence_frames)) / total_frames
            speech_detected = bool(speech_ratio >= 0.15 or (mean_rms >= self.speech_threshold_rms and mean_zcr >= 0.02))

            # 5. Sound Activity Pattern Identification
            if speech_detected and speech_ratio >= 0.45:
                activity_pattern = "active_discussion"
            elif speech_detected:
                activity_pattern = "intermittent_speech"
            elif mean_rms > 0.12 or noise_level_db > -18.0:
                activity_pattern = "high_ambient_disturbance"
            elif silence_ratio >= 0.70:
                activity_pattern = "quiet_listening"
            else:
                activity_pattern = "normal_ambient_noise"

            # 6. Normalized Acoustic Engagement Metric (0.0 to 1.0)
            # High speech participation contributes positively, excessive noise/disturbance penalized
            if activity_pattern == "active_discussion":
                engagement_score = min(1.0, 0.70 + speech_ratio * 0.30)
            elif activity_pattern == "intermittent_speech":
                engagement_score = min(0.75, 0.40 + speech_ratio * 0.35)
            elif activity_pattern == "quiet_listening":
                # Attentive quiet listening still represents valid engagement in lectures
                engagement_score = 0.50
            elif activity_pattern == "high_ambient_disturbance":
                engagement_score = 0.20
            else:
                engagement_score = 0.40

            return {
                "rms_energy": round(mean_rms, 5),
                "noise_level_db": noise_level_db,
                "speech_detected": speech_detected,
                "speech_ratio": round(speech_ratio, 3),
                "silence_ratio": round(silence_ratio, 3),
                "zero_crossing_rate": round(mean_zcr, 4),
                "spectral_centroid": round(mean_centroid, 2),
                "activity_pattern": activity_pattern,
                "acoustic_engagement_score": round(engagement_score, 3),
            }
        except Exception as e:
            print(f"Error analyzing audio chunk: {e}")
            return None

