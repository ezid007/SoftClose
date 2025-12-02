import cv2
import numpy as np
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

# Motion detection tuning (night noise reduction)
MOTION_VAR_THRESHOLD = int(
    os.getenv("VIDEO_MOTION_VAR_THRESHOLD", "40")
)  # MOG2 variance threshold (higher = less sensitive)
MOTION_HISTORY = int(os.getenv("VIDEO_MOTION_HISTORY", "500"))  # MOG2 history frames
MOTION_MIN_AREA = int(
    os.getenv("VIDEO_MOTION_MIN_AREA", "2000")
)  # Minimum contour area
MOTION_CONSECUTIVE_FRAMES = int(
    os.getenv("VIDEO_MOTION_CONSECUTIVE_FRAMES", "2")
)  # Required consecutive motion frames
TEMPORAL_SMOOTHING_ALPHA = float(
    os.getenv("VIDEO_TEMPORAL_ALPHA", "0.7")
)  # EMA smoothing (0-1, lower = more smoothing)

# YOLO detection tuning
YOLO_CONFIDENCE = float(
    os.getenv("VIDEO_YOLO_CONFIDENCE", "0.55")
)  # Detection confidence threshold
YOLO_MIN_BOX_AREA = int(
    os.getenv("VIDEO_YOLO_MIN_BOX_AREA", "3000")
)  # Minimum person box area in pixels
YOLO_OVERLAP_RATIO = float(
    os.getenv("VIDEO_YOLO_OVERLAP_RATIO", "0.2")
)  # Min intersection ratio with motion


class VideoRecorder:
    def __init__(self, audio_recorder=None):
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.out = None
        self.model = YOLO("yolo11n.pt")  # Load nano model for speed
        self.last_motion_time = 0
        self.recording_start_time = 0
        self.frame_buffer = []  # Pre-record buffer for smooth video start
        self.frame_buffer_size = 30  # ~1 second buffer at 30 FPS
        self.current_frame = None  # For GUI to access
        self.person_currently_detected = False  # Track if person is currently detected
        self.lock = threading.Lock()
        self.on_stop_callback = None
        self.on_person_detected_callback = None
        self.audio_recorder = (
            audio_recorder  # Reference to audio recorder for dB display
        )
        self.flip_horizontal = True  # Toggle for horizontal flip
        self.consecutive_person_frames = 0  # debounce counter
        self.required_person_frames = 3  # require N consecutive frames

        # Night noise reduction components
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOTION_HISTORY,
            varThreshold=MOTION_VAR_THRESHOLD,
            detectShadows=True,
        )
        self.temporal_smoothed_frame = None  # For EMA temporal smoothing
        self.consecutive_motion_frames = 0  # Motion persistence counter
        self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
        if ret and self.flip_horizontal:
            frame1 = cv2.flip(frame1, 1)

        while self.is_running and self.cap.isOpened():
            if not ret:
                break

            # ========================================
            # 1. Temporal Smoothing (Noise Reduction)
            # ========================================
            # Apply EMA (Exponential Moving Average) to reduce temporal noise
            frame_float = frame1.astype(np.float32)
            if self.temporal_smoothed_frame is None:
                self.temporal_smoothed_frame = frame_float.copy()
            else:
                # Alpha controls smoothing: lower = more smoothing but more blur on motion
                cv2.accumulateWeighted(
                    frame_float, self.temporal_smoothed_frame, TEMPORAL_SMOOTHING_ALPHA
                )

            smoothed_frame = self.temporal_smoothed_frame.astype(np.uint8)

            # ========================================
            # 2. Motion Detection with MOG2
            # ========================================
            # MOG2 learns background adaptively, much better for varying lighting/noise
            fg_mask = self.bg_subtractor.apply(smoothed_frame)

            # Remove shadows (gray pixels in MOG2 mask = 127)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            # Morphological operations to remove noise and fill gaps
            # Opening removes small noise, Closing fills small holes
            fg_mask = cv2.morphologyEx(
                fg_mask, cv2.MORPH_OPEN, self.morphology_kernel, iterations=2
            )
            fg_mask = cv2.morphologyEx(
                fg_mask, cv2.MORPH_CLOSE, self.morphology_kernel, iterations=2
            )
            fg_mask = cv2.dilate(fg_mask, self.morphology_kernel, iterations=1)

            # Find contours in the cleaned mask
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            frame_has_motion = False
            motion_bboxes = []
            total_motion_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MOTION_MIN_AREA:
                    frame_has_motion = True
                    total_motion_area += area
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_bboxes.append((x, y, x + w, y + h))

            # ========================================
            # 3. Motion Persistence Filter
            # ========================================
            # Require consecutive frames with motion to avoid single-frame noise spikes
            if frame_has_motion:
                self.consecutive_motion_frames += 1
            else:
                self.consecutive_motion_frames = 0

            motion_detected = (
                self.consecutive_motion_frames >= MOTION_CONSECUTIVE_FRAMES
            )

            person_detected = False
            annotated_frame = frame1.copy()

            # ========================================
            # 4. YOLO Detection (Only if motion detected or already recording)
            # ========================================
            if motion_detected or self.is_recording:
                results = self.model(
                    frame1, verbose=False, classes=[0], conf=YOLO_CONFIDENCE
                )

                # Check if any detection overlaps with motion region
                for result in results:
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            box_area = (x2 - x1) * (y2 - y1)

                            # Filter out tiny detections (likely noise)
                            if box_area < YOLO_MIN_BOX_AREA:
                                continue

                            # Calculate overlap ratio with motion regions
                            best_overlap_ratio = 0.0
                            for mx1, my1, mx2, my2 in motion_bboxes:
                                # Calculate intersection
                                ix1 = max(x1, mx1)
                                iy1 = max(y1, my1)
                                ix2 = min(x2, mx2)
                                iy2 = min(y2, my2)

                                if ix1 < ix2 and iy1 < iy2:
                                    intersection = (ix2 - ix1) * (iy2 - iy1)
                                    overlap_ratio = intersection / box_area
                                    best_overlap_ratio = max(
                                        best_overlap_ratio, overlap_ratio
                                    )

                            # Require minimum overlap ratio
                            if best_overlap_ratio >= YOLO_OVERLAP_RATIO:
                                person_detected = True
                                break

                        if person_detected:
                            annotated_frame = result.plot()  # Draw boxes
                            break

            # Recording Logic
            if person_detected:
                self.consecutive_person_frames += 1
                if self.consecutive_person_frames >= self.required_person_frames:
                    self.person_currently_detected = True
                    self.last_motion_time = time.time()
                    # Notify main app to start recording if not already
                    if self.on_person_detected_callback:
                        self.on_person_detected_callback()
            else:
                self.consecutive_person_frames = 0
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

            # Store frames in buffer for pre-recording
            if len(self.frame_buffer) >= self.frame_buffer_size:
                self.frame_buffer.pop(0)
            self.frame_buffer.append(annotated_frame.copy())

            if self.is_recording:
                self.out.write(annotated_frame)
                # Stop recording if no person for 5 seconds
                if time.time() - self.last_motion_time > 5:
                    self.stop_recording()

            # Update current frame for GUI
            with self.lock:
                self.current_frame = annotated_frame

            # Prepare for next iteration
            ret, frame1 = self.cap.read()
            if ret and self.flip_horizontal:
                frame1 = cv2.flip(frame1, 1)

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

        # Write pre-buffer frames for smooth start
        for buffered_frame in self.frame_buffer:
            self.out.write(buffered_frame)
        print(f"Wrote {len(self.frame_buffer)} pre-buffer frames")

        self.current_filename = filename
        self.current_filepath = filepath
        print(
            f"Started video recording at {self.real_start_time.strftime('%H:%M:%S.%f')}: {filename}"
        )

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

    def toggle_flip(self):
        """Toggle horizontal flip on/off."""
        self.flip_horizontal = not self.flip_horizontal
        status = "ON" if self.flip_horizontal else "OFF"
        print(f"Horizontal flip toggled: {status}")
        return self.flip_horizontal


if __name__ == "__main__":
    recorder = VideoRecorder()
    recorder.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
