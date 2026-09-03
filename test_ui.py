import cv2
import requests
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
import os

API_URL = "http://127.0.0.1:8000/api/v1"

class FaceApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x700")
        self.window.configure(bg="#F3F4F6") # Light gray background
        
        # Configure styles
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            pass
            
        style.configure('TFrame', background='#F3F4F6')
        style.configure('Card.TFrame', background='#FFFFFF', relief='flat')
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), background='#FFFFFF', foreground='#111827')
        style.configure('Subtitle.TLabel', font=('Segoe UI', 11), background='#FFFFFF', foreground='#6B7280')
        
        # Primary Button (Upload)
        style.configure('Primary.TButton', font=('Segoe UI', 11, 'bold'), padding=10)
        style.map('Primary.TButton', 
                  background=[('active', '#2563EB'), ('!active', '#3B82F6')], 
                  foreground=[('!active', 'white'), ('active', 'white')])

        # Success Button (Mark Attendance)
        style.configure('Success.TButton', font=('Segoe UI', 13, 'bold'), padding=12)
        style.map('Success.TButton', 
                  background=[('active', '#059669'), ('!active', '#10B981')], 
                  foreground=[('!active', 'white'), ('active', 'white')])

        # Main Container
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # --- LEFT PANE (Registration) ---
        self.pane_left = ttk.Frame(main_container, style='Card.TFrame')
        self.pane_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        ttk.Label(self.pane_left, text="Register Employee", style='Title.TLabel').pack(pady=(35, 5))
        ttk.Label(self.pane_left, text="Upload a clear photo to register a new face", style='Subtitle.TLabel').pack(pady=(0, 30))
        
        # Form Container
        form_frame = ttk.Frame(self.pane_left, style='Card.TFrame')
        form_frame.pack(fill=tk.X, padx=50)
        
        ttk.Label(form_frame, text="Employee ID", font=('Segoe UI', 10, 'bold'), background='#FFFFFF', foreground='#374151').pack(anchor='w', pady=(0, 5))
        
        self.emp_id_entry = ttk.Entry(form_frame, font=('Segoe UI', 12))
        self.emp_id_entry.pack(fill=tk.X, pady=(0, 25), ipady=6)
        
        self.btn_upload = ttk.Button(form_frame, text="Upload Photo & Register", style='Primary.TButton', command=self.register_via_upload)
        self.btn_upload.pack(fill=tk.X, pady=(0, 20))
        
        # Thumbnail preview area
        self.upload_img_label = tk.Label(self.pane_left, bg="#FFFFFF")
        self.upload_img_label.pack(pady=10)

        # --- RIGHT PANE (Live Attendance) ---
        self.pane_right = ttk.Frame(main_container, style='Card.TFrame')
        self.pane_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))
        
        ttk.Label(self.pane_right, text="Live Attendance", style='Title.TLabel').pack(pady=(35, 5))
        ttk.Label(self.pane_right, text="Look at the camera and mark attendance", style='Subtitle.TLabel').pack(pady=(0, 20))
        
        # Webcam Feed Container (with a nice border)
        cam_frame = tk.Frame(self.pane_right, bg="#E5E7EB", bd=0)
        cam_frame.pack(pady=10, padx=30)
        
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        
        if self.vid.isOpened():
            self.vid_width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.75)
            self.vid_height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.75)
        else:
            self.vid_width, self.vid_height = 480, 360
            
        self.canvas = tk.Canvas(cam_frame, width=self.vid_width, height=self.vid_height, bg="#111827", highlightthickness=4, highlightbackground="#E5E7EB")
        self.canvas.pack()
        
        self.btn_recognize = ttk.Button(self.pane_right, text="Mark Live Attendance", style='Success.TButton', command=self.mark_attendance)
        self.btn_recognize.pack(pady=25, padx=50, fill=tk.X)

        # Start video loop
        if self.vid.isOpened():
            self.delay = 15
            self.update_frame()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (self.vid_width, self.vid_height))
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)
            
    def update_frame(self):
        ret, frame = self.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update_frame)
        
    def get_image_bytes(self):
        ret, frame = self.vid.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        return None

    def register_via_upload(self):
        emp_id = self.emp_id_entry.get().strip()
        if not emp_id:
            messagebox.showerror("Error", "Please enter an Employee ID first.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Profile Picture",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
            
        try:
            img = Image.open(file_path)
            img.thumbnail((250, 250))
            thumb = ImageTk.PhotoImage(img)
            self.upload_img_label.config(image=thumb)
            self.upload_img_label.image = thumb
        except Exception as e:
            print(f"Could not load thumbnail: {e}")
            
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
                
            files = {'image': (os.path.basename(file_path), img_bytes, 'image/jpeg')}
            response = requests.post(f"{API_URL}/employees/{emp_id}/face", files=files)
            result = response.json()
            
            if response.status_code == 200 and result.get('success'):
                messagebox.showinfo("Success", f"Employee {emp_id} registered and saved to database successfully!")
                self.emp_id_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Error", result.get('detail', 'Unknown error occurred'))
        except requests.exceptions.ConnectionError:
            messagebox.showerror("Connection Error", "Is the FastAPI server running? (uvicorn main:app)")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def mark_attendance(self):
        img_bytes = self.get_image_bytes()
        if not img_bytes:
            messagebox.showerror("Error", "Could not capture image from webcam")
            return
            
        try:
            files = {'image': ('face.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(f"{API_URL}/attendance/recognize", files=files)
            result = response.json()
            
            if response.status_code == 200:
                if result.get('attendance_marked') or result.get('message') == 'Attendance already marked':
                    msg = f"Employee: {result.get('employee_id')}\n" \
                          f"Confidence: {result.get('confidence', 'N/A')}\n" \
                          f"Message: {result.get('message', 'Success')}"
                    messagebox.showinfo("Recognized!", msg)
                else:
                    conf = result.get('confidence', 'N/A')
                    messagebox.showwarning("Unknown", f"Face not recognized.\nBest match confidence: {conf}")
            else:
                messagebox.showerror("Error", result.get('detail', 'Unknown error occurred'))
        except requests.exceptions.ConnectionError:
            messagebox.showerror("Connection Error", "Is the FastAPI server running? (uvicorn main:app)")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root, "Face Registration & Attendance Portal")
    root.mainloop()
