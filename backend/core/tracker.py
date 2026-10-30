"""
Face and Participant Tracking Module
Associates detected faces across consecutive frames using spatial overlap (IoU)
and centroid movement patterns to maintain persistent participant IDs.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional

def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return float(intersection) / float(union)

def compute_centroid_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Computes Euclidean distance between the centroids of two bounding boxes."""
    c1 = (bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0)
    c2 = (bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0)
    return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

class TrackedParticipant:
    """Represents an individual participant tracked across frames."""
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        self.disappeared_frames = 0
        self.total_frames = 1
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.history: List[Tuple[float, float]] = [self.centroid]

    def update(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox
        self.centroid = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        self.disappeared_frames = 0
        self.total_frames += 1
        self.last_seen = time.time()
        self.history.append(self.centroid)
        if len(self.history) > 30:
            self.history.pop(0)

class FaceTracker:
    """
    Tracks detected faces across consecutive frames.
    Maintains identity consistency under head turns, movement, and brief occlusions.
    """
    def __init__(self, max_disappeared: int = 15, iou_threshold: float = 0.3, distance_threshold: float = 80.0):
        self.next_track_id = 1
        self.tracked_objects: Dict[int, TrackedParticipant] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold

    @property
    def tracks(self) -> Dict[int, TrackedParticipant]:
        """Convenience alias for tracked_objects."""
        return self.tracked_objects

    def register(self, bbox: Tuple[int, int, int, int]) -> int:
        track_id = self.next_track_id
        self.tracked_objects[track_id] = TrackedParticipant(track_id, bbox)
        self.next_track_id += 1
        return track_id

    def deregister(self, track_id: int):
        if track_id in self.tracked_objects:
            del self.tracked_objects[track_id]

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Updates the tracker with newly detected bounding boxes.
        Returns a structured list of tracked participants:
        [{"track_id": int, "bbox": (x, y, w, h), "centroid": (cx, cy), "timestamp": float}]
        """
        current_time = time.time()

        if len(detections) == 0:
            for track_id in list(self.tracked_objects.keys()):
                self.tracked_objects[track_id].disappeared_frames += 1
                if self.tracked_objects[track_id].disappeared_frames > self.max_disappeared:
                    self.deregister(track_id)
            return self._format_results(current_time)

        new_bboxes = [d["bbox"] for d in detections]

        if len(self.tracked_objects) == 0:
            for bbox in new_bboxes:
                self.register(bbox)
            return self._format_results(current_time)

        track_ids = list(self.tracked_objects.keys())
        existing_bboxes = [self.tracked_objects[tid].bbox for tid in track_ids]

        # Compute cost matrices: IoU and Centroid Distance
        iou_matrix = np.zeros((len(existing_bboxes), len(new_bboxes)), dtype=np.float32)
        dist_matrix = np.zeros((len(existing_bboxes), len(new_bboxes)), dtype=np.float32)

        for i, eb in enumerate(existing_bboxes):
            for j, nb in enumerate(new_bboxes):
                iou_matrix[i, j] = compute_iou(eb, nb)
                dist_matrix[i, j] = compute_centroid_distance(eb, nb)

        matched_tracks = set()
        matched_detections = set()

        # Step 1: Match by high spatial overlap (IoU)
        if len(iou_matrix) > 0 and len(iou_matrix[0]) > 0:
            for i in range(len(existing_bboxes)):
                best_det = int(np.argmax(iou_matrix[i]))
                best_iou = iou_matrix[i, best_det]
                if best_iou >= self.iou_threshold and best_det not in matched_detections:
                    tid = track_ids[i]
                    self.tracked_objects[tid].update(new_bboxes[best_det])
                    matched_tracks.add(i)
                    matched_detections.add(best_det)

        # Step 2: Fallback to spatial proximity (Centroid distance) for moving subjects
        for i in range(len(existing_bboxes)):
            if i in matched_tracks:
                continue
            for j in range(len(new_bboxes)):
                if j in matched_detections:
                    continue
                if dist_matrix[i, j] <= self.distance_threshold:
                    tid = track_ids[i]
                    self.tracked_objects[tid].update(new_bboxes[j])
                    matched_tracks.add(i)
                    matched_detections.add(j)
                    break

        # Step 3: Handle unmatched existing tracks
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracked_objects[tid].disappeared_frames += 1
                if self.tracked_objects[tid].disappeared_frames > self.max_disappeared:
                    self.deregister(tid)

        # Step 4: Register new detections as new participants
        for j, bbox in enumerate(new_bboxes):
            if j not in matched_detections:
                self.register(bbox)

        return self._format_results(current_time)

    def _format_results(self, timestamp: float) -> List[Dict]:
        results = []
        for tid, obj in self.tracked_objects.items():
            if obj.disappeared_frames == 0:  # actively detected in this frame
                results.append({
                    "track_id": tid,
                    "participant_id": f"participant_{tid}",
                    "bbox": obj.bbox,
                    "centroid": obj.centroid,
                    "timestamp": timestamp,
                    "total_frames": obj.total_frames,
                })
        return results
