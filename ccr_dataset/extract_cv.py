import json

import cv2
import os


def extract_frames(video_path, output_dir, fps=None, img_width=None):
    """
    Extracts frames from a video using OpenCV on Windows.

    Args:
            video_path (str): Path to the input video file.
            output_dir (str): Path to the directory where extracted frames will be saved.
                    If it doesn't exist, it will be created.
            fps (int, optional): Target frame rate for extracted frames.
                    Defaults to None (use original video FPS).
            img_width (int, optional): Target width for resized frames.
                    Defaults to None (no resizing).

    Returns:
            None
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video capture
    cap = cv2.VideoCapture(video_path)

        # # Get video properties (FPS, frame width)
        # original_fps = cap.get(cv2.CAP_PROP_FPS)
        # original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #
        # # Determine actual FPS and frame width to use
        # actual_fps = fps if fps is not None else original_fps
        # actual_width = img_width if img_width is not None else original_width
        #
        # # Define frame resizing if needed
        # resize_func = lambda frame: cv2.resize(frame, (actual_width, int(actual_width * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / original_width))) if img_width else lambda frame: frame

    # Extract frames
    frame_count = 0
    success, frame = cap.read()
    while success:
            # frame = resize_func(frame)    # Apply resizing if needed
            frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

            # Limit extraction based on FPS (if specified)
            if fps is not None:
                    wait_time = 1 / fps * 1000    # milliseconds
                    cv2.waitKey(int(wait_time))

            success, frame = cap.read()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    json_fid = open('config.json')
    config = json.load(json_fid)

    video_dir = os.path.join(config['dataset_dir'], 'videos')
    output_dir = os.path.join(config['dataset_dir'], 'frames')
    # Optional: Specify target FPS (e.g., fps=10) and/or target image width (e.g., img_width=320)
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                frame_dir = os.path.join(output_dir, os.path.splitext(file)[0])    # Create frame directory with same name as video (minus extension)
                extract_frames(video_path, frame_dir)    # Pass frame directory as output_dir