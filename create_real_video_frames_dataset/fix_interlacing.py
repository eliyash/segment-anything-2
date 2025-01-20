import subprocess
from pathlib import Path

import cv2
import numpy as np
import os

FFMPEG_PATH = r'C:\Portable\ffmpeg-4.4.1\bin\ffmpeg.exe'


def fix_interlacing_ffmpeg(video_path, output_path):
    # FFmpeg command to fix interlacing using yadif
    command = [
        FFMPEG_PATH,
        "-i", video_path,
        "-vf", "yadif",  # Apply the deinterlacing filter
        "-c:v", "libx264",  # Re-encode with H.264 codec
        "-preset", "fast",  # Encoding preset
        "-crf", "23",  # Constant Rate Factor (quality setting)
        "-c:a", "copy",  # Copy audio without re-encoding
        output_path
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Fixed video saved at: {output_path}")


def detect_interlacing_ffmpeg(video_path):
    """
    Detect interlacing artifacts in a video using FFmpeg's analysis filters.

    Parameters:
        video_path: Path to the input video.

    Returns:
        True if interlacing is detected, False otherwise.
    """
    command = [
        FFMPEG_PATH,
        "-i", video_path,
        "-vf", "idet",
        "-frames:v", "500",  # Analyze the first 500 frames (adjust as needed)
        "-f", "null", "-"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    analysis_output = result.stderr

    # Look for interlacing detection in the output log
    if "TFF:" in analysis_output or "BFF:" in analysis_output:
        print("Interlacing detected in the video.")
        return True
    else:
        print("No interlacing detected in the video.")
        return False


def temporal_deinterlace(current_frame, previous_frame):
    """
    Use temporal information to deinterlace the frame.

    Parameters:
        current_frame: Current video frame as a NumPy array.
        previous_frame: Previous video frame as a NumPy array.

    Returns:
        Deinterlaced frame as a NumPy array.
    """
    if previous_frame is None:
        return current_frame  # No previous frame to use

    # Combine odd rows from current frame and even rows from previous frame
    deinterlaced = np.zeros_like(current_frame)
    deinterlaced[0::2, :, :] = previous_frame[0::2, :, :]  # Even rows from previous
    deinterlaced[1::2, :, :] = current_frame[1::2, :, :]  # Odd rows from current

    return deinterlaced


def fix_interlacing(video_path, output_video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    previous_frame = None

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        # Fix interlacing
        fixed_frame = temporal_deinterlace(current_frame, previous_frame)

        cv2.imshow('current_frame', current_frame)
        cv2.imshow('fixed_frame', fixed_frame)
        cv2.waitKey()
        # Write to output video
        out.write(fixed_frame)

        # Update the previous frame
        previous_frame = current_frame

    # Release resources
    cap.release()
    out.release()

    print(f"Fixed video saved at: {output_video_path}")


def detect_interlacing(video_path, threshold=30):
    """
    Check if a video has interlacing artifacts.

    Parameters:
        video_path: Path to the input video.
        threshold: Difference threshold to detect interlacing artifacts.

    Returns:
        True if interlacing artifacts are detected, otherwise False.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    previous_frame = None
    interlacing_detected = False

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        if previous_frame is not None:
            # Compute absolute difference between frames
            diff = cv2.absdiff(current_frame, previous_frame)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Check for row-wise inconsistencies
            inconsistent_rows = np.mean(diff_gray, axis=1) > threshold
            if np.any(inconsistent_rows):
                interlacing_detected = True
                break

        # Update the previous frame
        previous_frame = current_frame

    cap.release()
    return interlacing_detected


def mian():
    videos_folder = Path(r'D:\per_signal_videos')
    output_folder = Path(r'D:\per_signal_videos_fixed_interlacing')
    output_folder.mkdir(exist_ok=True)

    for video_path in videos_folder.glob("*.mp4"):
        fixed_interlacing_path = output_folder / video_path.name
        if fixed_interlacing_path.exists():
            continue
        else:
            print("\tInterlacing artifacts detected. Fixing the video...")
            fix_interlacing_ffmpeg(video_path.as_posix(), fixed_interlacing_path.as_posix())

        # print(f"Checking: {video_path}")
        # has_interlacing = detect_interlacing_ffmpeg(video_path.as_posix())
        # if has_interlacing:
        #     print("\tInterlacing artifacts detected. Fixing the video...")
        #     fix_interlacing_ffmpeg(video_path.as_posix(), fixed_interlacing_path.as_posix())
        #     # fix_interlacing(video_path.as_posix(), (output_folder / video_path.name).as_posix())
        # else:
        #     print("\tNo interlacing artifacts detected.")


# Example usage
if __name__ == "__main__":
    mian()
