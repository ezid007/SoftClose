import os
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import subprocess

load_dotenv()


def ensure_smb_connection():
    """
    Ensure SMB connection to DB server is established.
    This should be called once at startup.
    """
    target_db_ip = os.getenv("DB_HOST", "192.168.0.192")
    target_db_share = os.getenv("TARGET_DB_SERVER_SHARE", "cctv_db")
    smb_username = os.getenv("SMB_USERNAME")
    smb_password = os.getenv("SMB_PASSWORD")

    smb_path = rf"\\{target_db_ip}\{target_db_share}"

    # Check if already connected
    if os.path.exists(smb_path):
        print(f"SMB connection already exists: {smb_path}")
        return True

    # Try to establish connection
    if smb_username and smb_password:
        print(f"Establishing SMB connection to {smb_path}...")
        cmd = f"net use {smb_path} /user:{smb_username} {smb_password}"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ SMB connection established: {smb_path}")
                return True
            else:
                print(f"Warning: Failed to establish SMB connection: {result.stderr}")
                return False
        except Exception as e:
            print(f"Warning: Error establishing SMB connection: {e}")
            return False
    else:
        print("Warning: SMB credentials not found in .env file")
        return False


def rename_merged_video(original_path, start_time):
    """
    Rename merged video file to YYYYMMDD_HHMMSS.avi format.

    Args:
        original_path: Original file path
        start_time: datetime object of recording start time

    Returns:
        New file path after renaming
    """
    try:
        # Get directory and extension
        directory = os.path.dirname(original_path)
        extension = os.path.splitext(original_path)[1]

        # Create new filename: YYYYMMDD_HHMMSS.avi
        new_filename = start_time.strftime("%Y%m%d_%H%M%S") + extension
        new_path = os.path.join(directory, new_filename)

        # Rename file
        os.rename(original_path, new_path)
        print(f"Renamed: {os.path.basename(original_path)} -> {new_filename}")

        return new_path
    except Exception as e:
        print(f"Error renaming file: {e}")
        return original_path


def copy_to_server(file_path, max_db, db_threshold=50.0, server_path=None):
    """
    Copy video file from CCTV server to DB server via SMB if dB level exceeds threshold.

    New Workflow:
    1. Check if dB >= threshold
    2. Move file to records/transferring/ folder
    3. Copy to DB server
    4. Move file to records/transferred/ folder

    This prevents duplicate transfers.

    Args:
        file_path: Path to the video file (in records/merged/)
        max_db: Maximum dB level recorded
        db_threshold: Minimum dB level to copy (default: 50.0)
        server_path: Target DB server destination path

    Returns:
        Final file path (transferred or merged folder)
    """
    if max_db < db_threshold:
        print(f"Skipping copy: dB level {max_db:.1f} is below threshold {db_threshold}")
        return file_path

    # Setup folder structure
    base_dir = os.path.dirname(os.path.dirname(file_path))  # records/
    transferring_dir = os.path.join(base_dir, "transferring")
    transferred_dir = os.path.join(base_dir, "transferred")

    # Create folders if they don't exist
    os.makedirs(transferring_dir, exist_ok=True)
    os.makedirs(transferred_dir, exist_ok=True)

    filename = os.path.basename(file_path)
    transferring_path = os.path.join(transferring_dir, filename)
    transferred_path = os.path.join(transferred_dir, filename)

    # Check if already transferred (file exists in transferred folder)
    if os.path.exists(transferred_path):
        print(f"File already transferred: {filename}")
        return transferred_path

    try:
        # Step 1: Move to transferring folder
        print(f"[Transfer] Moving to transferring: {filename}")
        shutil.move(file_path, transferring_path)

        # Step 2: Prepare DB server path with date folder (YYYYMMDD)
        if server_path is None:
            target_db_ip = os.getenv("DB_HOST", "192.168.0.192")
            target_db_share = os.getenv("TARGET_DB_SERVER_SHARE", "cctv_db")
            target_db_subfolder = os.getenv("TARGET_DB_SERVER_SUBFOLDER", "video")

            # Extract date from filename (YYYYMMDD_HHMMSS.avi)
            date_folder = filename[:8]  # Get YYYYMMDD from filename

            # Create path: \\192.168.0.192\cctv_db\video\20251128
            server_path = rf"\\{target_db_ip}\{target_db_share}\{target_db_subfolder}\{date_folder}"

        # Check if server path exists, create if needed
        if not os.path.exists(server_path):
            print(f"Creating date folder on DB server: {server_path}")
            try:
                os.makedirs(server_path, exist_ok=True)
                print(f"✓ Date folder created: {server_path}")
            except Exception as e:
                print(f"Failed to create server directory: {e}")
                # Move back to merged folder
                shutil.move(transferring_path, file_path)
                return file_path

        # Step 3: Copy to DB server (in date folder)
        destination = os.path.join(server_path, filename)
        print(f"[Transfer] Copying to DB server: {filename} (dB: {max_db:.1f})")
        print(f"  Source: {transferring_path}")
        print(f"  Destination: {destination}")

        shutil.copy2(transferring_path, destination)
        print(f"✓ Successfully copied to DB server")

        # Step 4: Move to transferred folder
        print(f"[Transfer] Moving to transferred: {filename}")
        shutil.move(transferring_path, transferred_path)
        print(f"✓ Transfer complete: {filename}")

        return transferred_path

    except Exception as e:
        print(f"✗ Error during transfer: {e}")

        # Try to recover: move back to merged folder if still in transferring
        if os.path.exists(transferring_path):
            try:
                shutil.move(transferring_path, file_path)
                print(f"Recovered file to merged folder: {filename}")
            except Exception as recover_error:
                print(f"Failed to recover file: {recover_error}")

        return file_path


def process_merged_video(
    original_path, start_time, max_db, db_threshold=50.0, server_path=None
):
    """
    Process merged video with new workflow:

    1. Rename file to YYYYMMDD_HHMMSS format in records/merged/
    2. If dB >= threshold:
       - Move to records/transferring/
       - Copy to DB server
       - Move to records/transferred/
    3. If dB < threshold: stays in records/merged/

    Args:
        original_path: Original merged video path (in records/merged/)
        start_time: datetime object of recording start time
        max_db: Maximum dB level recorded
        db_threshold: Minimum dB level to copy to DB server (default: 50.0)
        server_path: Target server destination path (optional)

    Returns:
        Final file path after processing
    """
    # Step 1: Rename file
    new_path = rename_merged_video(original_path, start_time)

    # Step 2: Copy to DB server if threshold met (handles folder moves)
    final_path = copy_to_server(new_path, max_db, db_threshold, server_path)

    return final_path
