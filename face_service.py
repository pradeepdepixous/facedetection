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

    def _validate_face_quality(self, image, face):
        # face contains 15 elements: x, y, w, h, ...
        x, y, w, h = face[0:4]
        
        # 1. Size Validation
        if w < 60 or h < 60:
            raise ValueError(f"Face is too small ({int(w)}x{int(h)}). Please move closer to the camera.")
        
        # 2. Blur Validation
        # Crop the face region to analyze blur
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        face_roi = image[y:y+h, x:x+w]
        
        if face_roi.size > 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < 80.0:  # Adjustable threshold
                raise ValueError(f"Image is too blurry (Quality score: {variance:.1f}). Please hold still and ensure good lighting.")

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
            raise ValueError("No face detected in the image.")

        if len(faces) > 1:
            raise ValueError("Multiple faces detected. Please ensure only one person is in the frame.")

        face = faces[0]

        # Quality Gate
        self._validate_face_quality(image, face)

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
        image_bytes: bytes
    ):
        embedding = self.get_embedding(image_bytes)

        # Retrieve all faces from SQLite
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id, embedding FROM faces")
            rows = cursor.fetchall()

        if not rows:
            return {
                "status": "EMPTY",
                "message": "No employees registered in the system."
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
            
            if score > best_score:
                best_score = float(score)
                best_employee = employee_id

        return {
            "status": "SUCCESS",
            "employee_id": best_employee,
            "confidence": round(best_score, 4)
        }