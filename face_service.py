import cv2
import numpy as np
import os
import sqlite3

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

        # Initialize SQLite database
        self.db_file = "face_database.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    employee_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            ''')
            conn.commit()

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

        # Store embedding in SQLite (store as bytes)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO faces (employee_id, embedding)
                VALUES (?, ?)
            ''', (employee_id, embedding.tobytes()))
            conn.commit()

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

        # Retrieve all faces from SQLite
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id, embedding FROM faces")
            rows = cursor.fetchall()

        if not rows:
            return {
                "matched": False,
                "message": "No employees registered"
            }

        best_employee = None
        best_score = -1

        for employee_id, stored_embedding_bytes in rows:
            # Reconstruct the numpy array from bytes
            stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32).reshape(1, 128)
            
            # OpenCV SFace match score (Cosine similarity)
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