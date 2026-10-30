import cv2
import os

class FaceDetector:
    def __init__(self):
        # Use OpenCV's built-in Haarcascade for face detection
        # This is more robust across Python versions than MediaPipe
        haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haarcascade_path):
            # Fallback to local directory
            local_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
            if os.path.exists(local_path):
                haarcascade_path = local_path

        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
        if self.face_cascade.empty():
            print("Error: Could not load face cascade xml")

    def process_frame(self, frame):
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # Real-World Lighting Robustness:
        # If low contrast / harsh shadow caused 0 detections, apply histogram equalization
        if len(faces) == 0:
            equalized = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(equalized, 1.1, 5, minSize=(30, 30))

        detections = []
        for (x, y, w, h) in faces:
            # Format to match previous output: bbox (x, y, w, h), score (1.0 placeholder)
            detections.append({"bbox": (int(x), int(y), int(w), int(h)), "score": 1.0})
        return detections
