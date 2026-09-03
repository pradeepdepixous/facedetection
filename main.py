from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import datetime

from face_service import FaceRecognitionService


app = FastAPI(
    title="Face Recognition Attendance API",
    version="1.0.0"
)

face_service = FaceRecognitionService()
attendance = {}


@app.get("/")
def health_check():

    return {
        "status": "running",
        "service": "face-recognition"
    }


@app.post("/api/v1/employees/{employee_id}/face")
async def register_face(
    employee_id: str,
    image: UploadFile = File(...)
):

    try:

        image_bytes = await image.read()

        result = face_service.register_employee(
            employee_id,
            image_bytes
        )

        return {
            "success": True,
            **result
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@app.post("/api/v1/face/recognize")
async def recognize_face(
    image: UploadFile = File(...)
):

    try:

        image_bytes = await image.read()

        result = face_service.recognize(
            image_bytes
        )

        return {
            "success": True,
            **result
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.post("/api/v1/attendance/recognize")
async def mark_attendance(
    image: UploadFile = File(...)
):

    try:

        image_bytes = await image.read()

        result = face_service.recognize(
            image_bytes
        )

        if not result["matched"]:

            return {
                "success": True,
                "attendance_marked": False,
                "message": "Unknown person",
                "confidence": result.get("confidence")
            }

        employee_id = result["employee_id"]

        today = datetime.now().date().isoformat()

        key = f"{employee_id}:{today}"

        # Prevent duplicate attendance
        if key in attendance:

            return {
                "success": True,
                "attendance_marked": False,
                "message": "Attendance already marked",
                "employee_id": employee_id,
                "time": attendance[key]
            }

        current_time = datetime.now().isoformat()

        attendance[key] = current_time

        return {
            "success": True,
            "attendance_marked": True,
            "employee_id": employee_id,
            "confidence": result.get("confidence"),
            "time": current_time
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
