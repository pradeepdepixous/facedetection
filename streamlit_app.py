import streamlit as st
from face_service import FaceRecognitionService
from datetime import datetime
import os

st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("🧑‍💻 Face Recognition Attendance Portal")
st.markdown("---")

# Cache the service so the ML models only load once per server start
@st.cache_resource
def get_face_service():
    return FaceRecognitionService()

# Cache the attendance records in session state
if 'attendance' not in st.session_state:
    st.session_state.attendance = {}

try:
    face_service = get_face_service()
except Exception as e:
    st.error(f"Failed to load AI Models. Please ensure ONNX files are in the directory. Error: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.header("1. Register Employee")
    st.markdown("Register a new face in the database using your webcam or by uploading a photo.")
    
    emp_id = st.text_input("Employee ID (e.g., EMP001)", key="reg_id")
    
    tab1, tab2 = st.tabs(["📸 Use Webcam", "📁 Upload Photo"])
    
    img_bytes = None
    with tab1:
        st.info("""
        **📸 Camera Rules:**
        1. Ensure your face is well-lit (no strong backlight).
        2. Look straight into the camera.
        3. Keep your face centered in the frame.
        4. Remove masks or dark sunglasses.
        """)
        reg_camera = st.camera_input("Take a clear picture of your face", key="reg_cam")
        if reg_camera:
            img_bytes = reg_camera.getvalue()
            
    with tab2:
        reg_upload = st.file_uploader("Upload a face photo", type=['jpg', 'jpeg', 'png'])
        if reg_upload:
            img_bytes = reg_upload.getvalue()
    
    if st.button("Register Face", type="primary", use_container_width=True):
        if not emp_id:
            st.error("Please enter an Employee ID.")
        elif not img_bytes:
            st.error("Please provide an image (either take a photo or upload one).")
        else:
            with st.spinner('Registering face...'):
                try:
                    result = face_service.register_employee(emp_id, img_bytes)
                    st.success(f"✅ Successfully registered Employee: **{result['employee_id']}**")
                except ValueError as ve:
                    st.error(f"Validation Error: {ve}")
                except Exception as e:
                    st.error(f"Internal Error: {e}")

with col2:
    st.header("2. Live Attendance")
    st.markdown("Look at your webcam and take a picture to mark today's attendance.")
    
    st.info("""
    **📸 Camera Rules:**
    1. Ensure your face is well-lit (no strong backlight).
    2. Look straight into the camera.
    3. Keep your face centered in the frame.
    4. Remove masks or dark sunglasses.
    """)
    
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        if st.button("Mark Attendance", type="primary"):
            with st.spinner('Analyzing face...'):
                try:
                    img_bytes = camera_image.getvalue()
                    result = face_service.recognize(img_bytes)
                    
                    if result.get("status") == "EMPTY":
                        st.error(result["message"])
                    else:
                        employee_id = result["employee_id"]
                        confidence = result["confidence"]
                        
                        # 3-STATE LOGIC
                        if confidence >= 0.38:
                            # 1. MATCH - High Confidence
                            today = datetime.now().date().isoformat()
                            key = f"{employee_id}:{today}"
                            
                            if key in st.session_state.attendance:
                                st.info(f"Attendance already marked for today at {st.session_state.attendance[key]}")
                                st.success(f"✅ Recognized: **{employee_id}**")
                                st.caption(f"Confidence Score: {confidence}")
                            else:
                                current_time = datetime.now().isoformat()
                                st.session_state.attendance[key] = current_time
                                
                                st.success(f"✅ Recognized & Attendance Marked: **{employee_id}**")
                                st.info("Message: Attendance successfully recorded.")
                                st.caption(f"Confidence Score: {confidence}")
                                
                        elif 0.28 <= confidence < 0.38:
                            # 2. RETRY - Borderline Confidence
                            st.warning(f"⚠️ Borderline Match (Score: {confidence}). Please try again.")
                            st.markdown("- Move into better lighting\n- Look straight at the camera\n- Ensure your face is clearly visible")
                            
                        else:
                            # 3. REJECT - Low Confidence
                            st.error(f"❌ Unknown Face Rejected. (Highest Score: {confidence})")
                            
                except ValueError as ve:
                    st.error(f"Validation Error: {ve}")
                except Exception as e:
                    st.error(f"Internal Error: {e}")
