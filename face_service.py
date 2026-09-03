import cv2
import numpy as np
import os
import pickle

class FaceRecognitionService:

    def __init__(self):
        # Initialize OpenCV Face Detector (YuNet)
        self.detector = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx",
            "",
            (640, 640),
            0.9,
            0.3,
            5000
        )
        
        # Initialize OpenCV Face Recognizer (SFace)
        self.recognizer = cv2.FaceRecognizerSF.create(
            "face_recognition_sface_2021dec.onnx",
            ""
        )

        # POC storage backed by local file
        self.db_file = "embeddings_db.pkl"
        self.embeddings = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _save_db(self):
        with open(self.db_file, "wb") as f:
            pickle.dump(self.embeddings, f)

    def get_embedding(self, image_bytes: bytes):
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image")

        # Set input size to actual image size for YuNet
        height, width, _ = image.shape
        self.detector.setInputSize((width, height))

        # Detect faces
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            raise ValueError("No face detected")

        if len(faces) > 1:
            raise ValueError("Multiple faces detected. Please provide one face.")

        face = faces[0]

        # Align face based on landmarks
        aligned_face = self.recognizer.alignCrop(image, face)
        
        # Extract features/embedding
        embedding = self.recognizer.feature(aligned_face)

        return embedding

    def register_employee(
        self,
        employee_id: str,
        image_bytes: bytes
    ):
        embedding = self.get_embedding(image_bytes)

        self.embeddings[employee_id] = embedding
        self._save_db()

        # Save image to db/folder
        os.makedirs("employees", exist_ok=True)
        with open(os.path.join("employees", f"{employee_id}.jpg"), "wb") as f:
            f.write(image_bytes)

        return {
            "employee_id": employee_id,
            "message": "Face registered successfully"
        }

    def recognize(
        self,
        image_bytes: bytes,
        threshold: float = 0.36
    ):
        embedding = self.get_embedding(image_bytes)

        if not self.embeddings:
            return {
                "matched": False,
                "message": "No employees registered"
            }

        best_employee = None
        best_score = -1

        for employee_id, stored_embedding in self.embeddings.items():
            # OpenCV SFace match score (Cosine similarity)
            # 0 means Cosine distance type
            score = self.recognizer.match(
                embedding, stored_embedding, 0
            )
            
            # For Cosine in OpenCV SFace, >= 0.363 is generally considered a match
            if score > best_score:
                best_score = float(score)
                best_employee = employee_id

        if best_score >= threshold:
            return {
                "matched": True,
                "employee_id": best_employee,
                "confidence": round(best_score, 4)
            }

        return {
            "matched": False,
            "employee_id": None,
            "confidence": round(best_score, 4)
        }