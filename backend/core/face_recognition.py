from deepface import DeepFace
import cv2
import numpy as np
import os
import shutil

class FaceRecognizer:
    def __init__(self, db_path="student_db"):
        self.db_path = db_path
        if not os.path.exists(db_path):
            os.makedirs(db_path)

    def register_student(self, img_path, student_id):
        """
        Registers a student face by storing their reference image in the db_path folder.
        Supports both image filepath strings and numpy image arrays.
        """
        target_path = os.path.join(self.db_path, f"{student_id}.jpg")
        if isinstance(img_path, str) and os.path.exists(img_path):
            shutil.copy(img_path, target_path)
            return True
        elif isinstance(img_path, np.ndarray):
            cv2.imwrite(target_path, img_path)
            return True
        return False

    def recognize(self, frame):
        """
        Recognizes faces in the provided frame against registered student images.
        Returns a list of matched student records:
        [{"student_id": str, "confidence": float, "bbox": (x, y, w, h)}]
        """
        try:
            # Save frame temporarily or pass numpy array directly if supported (DeepFace supports numpy)
            # DeepFace.find requires a db_path with images.
            # If no DB yet, we can't recognize.
            if not os.listdir(self.db_path):
                return []

            # This is computing intensive, so we should run it sparingly
            dfs = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                silent=True,
                enforce_detection=False
            )

            results = []
            if isinstance(dfs, list):
                for df in dfs:
                    if hasattr(df, "empty") and not df.empty:
                        top_match = df.iloc[0]
                        identity_path = top_match["identity"]
                        # Extract student ID from filename (e.g., student_101.jpg -> student_101)
                        student_id = os.path.splitext(os.path.basename(identity_path))[0]
                        distance = float(top_match.get("distance", 0.0))
                        threshold = float(top_match.get("threshold", 0.6))

                        # Bounding box of the recognized face in the current frame
                        x = int(top_match.get("source_x", 0))
                        y = int(top_match.get("source_y", 0))
                        w = int(top_match.get("source_w", 0))
                        h = int(top_match.get("source_h", 0))

                        # Higher confidence if distance is significantly below threshold
                        confidence = max(0.0, min(1.0, 1.0 - (distance / max(threshold, 0.001))))

                        results.append({
                            "student_id": student_id,
                            "confidence": round(confidence, 3),
                            "distance": round(distance, 4),
                            "bbox": (x, y, w, h),
                        })
            return results
        except Exception as e:
            # Fail gracefully on empty detections or embedding computation issues
            return []
