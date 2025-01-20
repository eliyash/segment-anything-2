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
    fixed_interlacing_folder = Path(r'D:\per_signal_videos_fixed_interlacing')

    # for video_path in fixed_interlacing_folder.glob("*.mp4"):
    for video_path in fixed_interlacing_folder.glob("6_28_2018_20__index_500.mp4"):
        orig_video = videos_folder / video_path.name
        cap_fixed_interlacing = cv2.VideoCapture(video_path.as_posix())
        cap_orig_video = cv2.VideoCapture(orig_video.as_posix())
        if not cap_fixed_interlacing.isOpened() or not cap_orig_video.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        while True:
            ret, frame_fixed_interlacing = cap_fixed_interlacing.read()
            if not ret:
                break
            ret, frame_orig_video = cap_orig_video.read()
            if not ret:
                break

            combined_frame = np.hstack((frame_orig_video[:, :300], frame_fixed_interlacing[:, :300]))
            cv2.imshow('combined_frame', combined_frame)
            cv2.waitKey(200)


# Example usage
if __name__ == "__main__":
    mian()
