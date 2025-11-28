from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading
import time
import cv2
from contextlib import asynccontextmanager
from db_manager import init_db
from audio_recorder import AudioRecorder
from video_recorder import VideoRecorder
import logging
from moviepy import VideoFileClip, AudioFileClip
import os


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return message.find("/status") == -1 and message.find("/audio_waveform") == -1


# Global instances
audio_rec = AudioRecorder()
video_rec = VideoRecorder(audio_recorder=audio_rec)

# Shared state for dB level (simplified)
current_db = 0


def update_db_loop():
    global current_db
    while True:
        if audio_rec.is_running:
            current_db = int(audio_rec.current_db)
        else:
            current_db = 0
        time.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")

    # Filter out /status logs
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

    init_db()

    # Start recorders by default
    audio_rec.start()
    video_rec.start()

    # Link Audio Trigger to Video Recording
    # Link Audio Trigger to Video Recording
    # Link Audio Trigger to Video Recording (Extension Only)
    def trigger_video_recording(db_level):
        # Only extend recording if already running AND person is detected
        if video_rec.is_recording and video_rec.person_currently_detected:
            video_rec.last_motion_time = time.time()
            # print(f"Audio trigger: Extending recording (Level: {db_level:.1f}dB)")
        else:
            # Do NOT start recording on audio alone anymore
            pass

    audio_rec.set_trigger_callback(trigger_video_recording)

    # Link Person Detection to Recording Start
    def start_recording_on_person():
        # Start Video if not recording
        if not video_rec.is_recording:
            print("Person detected! Starting video and audio recording...")
            video_rec.start_recording()

            # Force start audio recording even if quiet
            if not audio_rec.is_recording:
                audio_rec.start_recording(db_level=0)  # 0dB as placeholder
        else:
            # Extend video recording
            video_rec.last_motion_time = time.time()

    video_rec.set_person_detected_callback(start_recording_on_person)

    def merge_recordings(video_path):
        """Callback when video recording stops. Stops audio and merges files."""
        print(f"Video stopped: {video_path}. Stopping audio and merging...")

        # Get start and end time from video recorder
        start_time = (
            video_rec.real_start_time
            if hasattr(video_rec, "real_start_time")
            else datetime.now()
        )
        end_time = (
            video_rec.real_end_time
            if hasattr(video_rec, "real_end_time")
            else datetime.now()
        )

        # Stop audio and get path
        audio_path = audio_rec.stop_recording()

        if not audio_path or not os.path.exists(audio_path):
            print("No audio file to merge. (Maybe recording was too short or failed)")
            # If no audio, just rename video to merged folder or leave it?
            # Let's just leave the video as is in records/video
            return

        try:
            # Merge using moviepy
            print(f"Merging {video_path} and {audio_path}...")

            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)

            # Trim audio to match video duration if needed
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclipped(0, video_clip.duration)

            final_clip = video_clip.with_audio(audio_clip)

            # Save merged file
            merge_dir = os.path.abspath(os.path.join("records", "merged"))
            os.makedirs(merge_dir, exist_ok=True)

            output_filename = f"merged_{os.path.basename(video_path)}"
            output_path = os.path.join(merge_dir, output_filename)

            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

            # Close clips to release files
            video_clip.close()
            audio_clip.close()

            print(f"Merge complete: {output_path}")

            # Log merged file to database
            from video_recorder import WIDTH, HEIGHT, FPS
            from db_manager import log_event

            metadata = {"width": WIDTH, "height": HEIGHT, "fps": FPS}

            log_event(
                f"{audio_rec.max_db_recorded:.1f}dB",
                output_path,
                start_time,
                end_time,
                metadata,
            )
            print(f"Logged to database: {output_path}")

            # Delete original files after successful merge
            try:
                os.remove(video_path)
                os.remove(audio_path)
                print(
                    f"Deleted originals: {os.path.basename(video_path)}, {os.path.basename(audio_path)}"
                )
            except Exception as e:
                print(f"Failed to delete originals: {e}")

        except Exception as e:
            print(f"Error merging files: {e}")

    video_rec.set_stop_callback(merge_recordings)

    # Start dB update thread
    threading.Thread(target=update_db_loop, daemon=True).start()

    yield

    # Shutdown
    print("Shutting down... Saving recordings...")

    # Disable callbacks to prevent triggers during shutdown
    audio_rec.set_trigger_callback(None)
    video_rec.set_person_detected_callback(None)
    video_rec.set_stop_callback(None)

    video_rec.stop()
    audio_rec.stop()


app = FastAPI(lifespan=lifespan)

# Setup templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def gen_frames():
    while True:
        frame = video_rec.get_frame()
        if frame:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            time.sleep(0.1)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
async def status():
    return JSONResponse({"db": current_db, "is_running": video_rec.is_running})


@app.post("/toggle_power")
async def toggle_power():
    if video_rec.is_running:
        video_rec.stop()
        audio_rec.stop()
        status = "OFF"
    else:
        video_rec.start()
        audio_rec.start()
        status = "ON"
    return JSONResponse({"status": status})


@app.get("/audio_waveform")
async def audio_waveform():
    """Returns recent audio waveform data for visualization."""
    if audio_rec.is_running and len(audio_rec.frames) > 0:
        try:
            # Get the latest frame
            import numpy as np

            data = audio_rec.frames[-1]
            # Convert to int16 array
            samples = np.frombuffer(data, dtype=np.int16)
            # Downsample to ~100 points for visualization
            step = max(1, len(samples) // 100)
            downsampled = samples[::step].tolist()
            return JSONResponse({"waveform": downsampled})
        except:
            return JSONResponse({"waveform": []})
    return JSONResponse({"waveform": []})


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    host = os.getenv("HOST")
    port = int(os.getenv("PORT"))

    uvicorn.run(app, host=host, port=port)
