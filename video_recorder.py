import cv2
import threading
import time
import os
from datetime import datetime
from ultralytics import YOLO
from db_manager import log_event
from dotenv import load_dotenv

load_dotenv()

# Config
WIDTH = int(os.getenv("VIDEO_WIDTH"))
HEIGHT = int(os.getenv("VIDEO_HEIGHT"))
FPS = float(os.getenv("VIDEO_FPS"))
RECORD_DIR = os.path.abspath(os.path.join("records", "video"))


class VideoRecorder:
    def __init__(self, audio_recorder=None):
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.out = None
        self.model = YOLO("yolo11n.pt")  # Load nano model for speed
        self.last_motion_time = 0
        self.recording_start_time = 0
        self.frame_buffer = (
            []
        )  # Optional: Pre-record buffer if needed, but user didn't explicitly ask for video pre-record, only audio.
        self.current_frame = None  # For GUI to access
        self.person_currently_detected = False  # Track if person is currently detected
        self.lock = threading.Lock()
        self.on_stop_callback = None
        self.on_person_detected_callback = None
        self.audio_recorder = (
            audio_recorder  # Reference to audio recorder for dB display
        )

    def set_stop_callback(self, callback):
        self.on_stop_callback = callback

    def set_person_detected_callback(self, callback):
        self.on_person_detected_callback = callback

    def start(self):
        """Starts the video monitoring in a separate thread."""
        if self.is_running:
            return

        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        threading.Thread(target=self.monitor_loop, daemon=True).start()
        print("Video monitoring started.")

    def stop(self):
        """Stops the video monitoring."""
        self.is_running = False
        if self.is_recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()

    def monitor_loop(self):
        """Main loop for video processing."""
        ret, frame1 = self.cap.read()
        if ret:
            frame1 = cv2.flip(frame1, 1)
        ret, frame2 = self.cap.read()
        if ret:
            frame2 = cv2.flip(frame2, 1)

        while self.is_running and self.cap.isOpened():
            if not ret:
                break

            # 1. Motion Detection
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > 900:  # Sensitivity threshold
                    motion_detected = True
                    break

            person_detected = False
            annotated_frame = frame1.copy()

            # 2. YOLO Detection (Only if motion detected or already recording)
            if motion_detected or self.is_recording:
                results = self.model(frame1, verbose=False, classes=[0])  # 0 is person

                # Check if person is detected
                for result in results:
                    if len(result.boxes) > 0:
                        person_detected = True
                        annotated_frame = result.plot()  # Draw boxes
                        break

            # Recording Logic
            if person_detected:
                self.person_currently_detected = True
                self.last_motion_time = time.time()

                # Notify main app to start recording if not already
                if self.on_person_detected_callback:
                    self.on_person_detected_callback()
            else:
                self.person_currently_detected = False

            # Add dB overlay to frame (BEFORE writing to file)
            if self.audio_recorder:
                db_text = f"{self.audio_recorder.current_db:.1f} dB"
                cv2.putText(
                    annotated_frame,
                    db_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if self.is_recording:
                self.out.write(annotated_frame)
                # Stop recording if no person for 5 seconds
                if time.time() - self.last_motion_time > 5:
                    self.stop_recording()

            # Update current frame for GUI
            with self.lock:
                self.current_frame = annotated_frame

            # Prepare for next iteration
            frame1 = frame2
            ret, frame2 = self.cap.read()
            if ret:
                frame2 = cv2.flip(frame2, 1)

            # Small sleep to match FPS if needed, but processing usually takes time
            # time.sleep(1/FPS)

    def start_recording(self):
        self.is_recording = True
        self.last_motion_time = time.time()  # Reset timer on start
        self.real_start_time = datetime.now()

        # Ensure directory exists
        os.makedirs(RECORD_DIR, exist_ok=True)

        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        filepath = os.path.join(RECORD_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(filepath, fourcc, FPS, (WIDTH, HEIGHT))

        if not self.out.isOpened():
            print(f"Error: Could not open video writer for {filename}")
            self.out = None
            self.is_recording = False
            return

        self.current_filename = filename
        self.current_filepath = filepath
        print(f"Started video recording: {filename}")

    def stop_recording(self):
        self.is_recording = False
        if self.out:
            self.out.release()
            self.out = None
            self.real_end_time = datetime.now()  # Record actual end time
            print(f"Stopped video recording.")

            # Callback will be triggered below (merge will handle DB logging)

            if self.on_stop_callback:
                self.on_stop_callback(self.current_filepath)

    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                ret, buffer = cv2.imencode(".jpg", self.current_frame)
                return buffer.tobytes()
            return None


if __name__ == "__main__":
    recorder = VideoRecorder()
    recorder.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
