import pyaudio
import wave
import numpy as np
from collections import deque
import threading
import time
import os
from datetime import datetime, timedelta
from db_manager import log_event
from dotenv import load_dotenv

load_dotenv()

# Config
CHUNK = int(os.getenv("AUDIO_CHUNK_SIZE"))
FORMAT = pyaudio.paInt16
CHANNELS = int(os.getenv("AUDIO_CHANNELS"))
RATE = int(os.getenv("AUDIO_RATE"))
THRESHOLD = int(os.getenv("AUDIO_THRESHOLD_DB"))
BUFFER_SECONDS = int(os.getenv("AUDIO_BUFFER_SECONDS"))
RECORD_SECONDS = int(os.getenv("AUDIO_RECORD_SECONDS"))
DB_OFFSET = int(os.getenv("AUDIO_DB_OFFSET"))


class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = deque(maxlen=int(RATE / CHUNK * BUFFER_SECONDS))
        self.is_running = False
        self.is_recording = False
        self.recording_frames = []
        self.remaining_chunks = 0
        self.trigger_desc = ""
        self.on_trigger_callback = None
        self.current_db = 0
        self.max_db_recorded = 0

    def set_trigger_callback(self, callback):
        """Sets the callback function to be called when audio threshold is exceeded."""
        self.on_trigger_callback = callback

    def calculate_levels(self, audio_data):
        """Calculates both RMS (average) and Peak (max) decibel levels."""
        try:
            data = np.frombuffer(audio_data, dtype=np.int16)

            # Reference: maximum value for int16
            reference = 32768.0

            # RMS Calculation
            rms = np.sqrt(np.mean(data**2))
            rms_db = 20 * np.log10(rms / reference) if rms > 0 else -96

            # Peak Calculation
            peak = np.max(np.abs(data))
            peak_db = 20 * np.log10(peak / reference) if peak > 0 else -96

            # Convert to positive scale (0 dB = silence, 96 dB = max)
            # Add 96 to shift from [-96, 0] range to [0, 96] range
            # Then add DB_OFFSET for calibration (default 0)
            rms_db_shifted = max(0, rms_db + 96 + DB_OFFSET)
            peak_db_shifted = max(0, peak_db + 96 + DB_OFFSET)

            return rms_db_shifted, peak_db_shifted
        except Exception:
            return 0, 0

    def calculate_db(self, audio_data):
        """Legacy wrapper for backward compatibility, returns max of RMS and Peak."""
        rms, peak = self.calculate_levels(audio_data)
        return max(rms, peak)

    def start(self):
        """Starts the audio monitoring in a separate thread."""
        if self.is_running:
            return

        self.is_running = True
        device_index = os.getenv("AUDIO_DEVICE_INDEX")
        if device_index is not None:
            device_index = int(device_index)

        # List available audio devices for debugging
        print("\n=== Available Audio Input Devices ===")
        try:
            info = self.p.get_host_api_info_by_index(0)
            numdevices = info.get("deviceCount")
            for i in range(0, numdevices):
                dev_info = self.p.get_device_info_by_host_api_device_index(0, i)
                if dev_info.get("maxInputChannels") > 0:
                    print(
                        f"  Device {i}: {dev_info.get('name')} (Max Channels: {dev_info.get('maxInputChannels')})"
                    )
        except Exception as e:
            print(f"Error listing devices: {e}")
        print("=====================================\n")

        # Try to open audio stream with fallback logic
        channels_to_try = [CHANNELS]
        if CHANNELS == 2:
            channels_to_try.append(1)  # Fallback to mono if stereo fails

        for channels in channels_to_try:
            try:
                # Build kwargs for p.open()
                stream_params = {
                    "format": FORMAT,
                    "channels": channels,
                    "rate": RATE,
                    "input": True,
                    "frames_per_buffer": CHUNK,
                }

                # Only add input_device_index if explicitly specified
                if device_index is not None:
                    stream_params["input_device_index"] = device_index
                    print(
                        f"Attempting to open audio: Device={device_index}, Channels={channels}, Rate={RATE}"
                    )
                else:
                    print(
                        f"Attempting to open audio: Default Device, Channels={channels}, Rate={RATE}"
                    )

                self.stream = self.p.open(**stream_params)
                threading.Thread(target=self.monitor_loop, daemon=True).start()
                print(
                    f"✓ Audio monitoring started successfully with {channels} channel(s)."
                )
                return  # Success!

            except Exception as e:
                print(f"✗ Failed with {channels} channel(s): {e}")
                if channels == channels_to_try[-1]:  # Last attempt
                    print(f"\n!!! All audio configuration attempts failed !!!")
                    print(f"Error: {e}")
                    self.is_running = False

    def stop(self):
        """Stops the audio monitoring."""
        self.is_running = False
        self.on_trigger_callback = None  # Disable callback

        # Save current recording if in progress
        if self.is_recording:
            print("Saving current audio recording before stopping...")
            self.save_recording()
            self.is_recording = False

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        self.p.terminate()
        print("Audio monitoring stopped.")

    def monitor_loop(self):
        """Main loop for monitoring audio and triggering recording."""
        print(f"Listening for sounds above {THRESHOLD}dB (RMS or Peak)...")
        while self.is_running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)

                if self.is_recording:
                    self.recording_frames.append(data)

                    # Update current dB level and track peak
                    rms, peak = self.calculate_levels(data)
                    max_db = max(rms, peak)
                    self.current_db = max_db
                    self.max_db_recorded = max(self.max_db_recorded, max_db)

                    # No auto-stop: recording continues until externally stopped via stop_recording()
                else:
                    rms, peak = self.calculate_levels(data)
                    max_db = max(rms, peak)
                    self.current_db = max_db

                    if max_db > THRESHOLD:
                        # Notify callback with dB level
                        # Do NOT start recording here automatically.
                        # The main application will decide based on video conditions.
                        if self.on_trigger_callback:
                            self.on_trigger_callback(max_db)

            except Exception as e:
                print(f"Error in audio loop: {e}")
                break

    def start_recording(self, db_level, start_time=None):
        """Initiates the recording process at exact timestamp."""
        if self.is_recording:
            return
        self.recording_start_time = start_time if start_time else datetime.now()
        print(
            f"DEBUG: start_recording called at {self.recording_start_time.strftime('%H:%M:%S.%f')}"
        )
        self.is_recording = True
        self.max_db_recorded = 0  # Reset peak dB for new recording
        self.recording_frames = []  # Start fresh - no pre-buffer for perfect sync
        print(f"DEBUG: Started recording (No pre-buffer for exact sync)")
        self.remaining_chunks = int(RATE / CHUNK * RECORD_SECONDS)
        self.trigger_desc = f"{db_level:.1f}dB Noise Detected"

    def stop_recording(self):
        """Stops the current recording and returns the saved file path."""
        if not self.is_recording:
            return None

        print("Stopping audio recording manually...")
        self.is_recording = False
        return self.save_recording()

    def save_recording(self):
        """Saves the recorded frames to a WAV file and logs the event."""
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        # Ensure directory exists
        save_dir = os.path.abspath(os.path.join("records", "audio"))
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        try:
            wf = wave.open(filepath, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.recording_frames))
            wf.close()

            print(
                f"Saved audio: {filepath} (Frames: {len(self.recording_frames)}, Peak: {self.max_db_recorded:.1f}dB)"
            )
            return filepath

        except Exception as e:
            print(f"Error saving audio file: {e}")
            return None


if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
