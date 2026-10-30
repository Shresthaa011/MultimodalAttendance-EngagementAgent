"""
Core modules for Multimodal Attendance & Engagement Agent.
Exposes Vision, Audio, and Multimodal Agent services.
"""

from .vision_service import VisionService
from .audio_processor import AudioProcessor
from .audio_analysis import AudioAnalyzer
from .multimodal_agent import MultimodalAgent
from .llm_insights import InsightGenerator
from .multimodal_fusion import MultimodalSynchronizer
from .ai_suggestions import AISuggestionEngine
from .engagement_scorer import EngagementScorer
from .teacher_profiles import TeacherProfileService
from .recommendations_engine import RecommendationsEngine
from .analytics_service import AnalyticsService

__all__ = [
    "VisionService",
    "AudioProcessor",
    "AudioAnalyzer",
    "MultimodalAgent",
    "InsightGenerator",
    "MultimodalSynchronizer",
    "AISuggestionEngine",
    "EngagementScorer",
    "TeacherProfileService",
    "RecommendationsEngine",
    "AnalyticsService",
]
