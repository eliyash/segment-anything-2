import json
from pathlib import Path

from florence_2.check_missing_lines import match_videos_to_signals

import cv2


def save_video_parts(video_path, out_path, param_dicts):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for param_dict in param_dicts:
        start_seconds = param_dict['start_seconds']
        end_seconds = param_dict['end_seconds']
        index_in_sheet = param_dict['index_in_sheet']
        key = '_'.join(map(str, param_dict['key']))

        # Define the output file name
        file_stem = f'{key}__index_{index_in_sheet}'
        video_out_path = out_path / f'{file_stem}.mp4'
        metadata_out_path = out_path / f'{file_stem}.json'

        if video_out_path.exists() and metadata_out_path.exists():
            # print(f"Skipping video part: {video_out_path}")
            continue
        # Open the VideoWriter to save the part
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out_path.as_posix(), fourcc, fps, (width, height))

        # Calculate start and end frames
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)

        # Set the starting frame of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Loop over the frames and save the part
        for frame_no in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        # Release the writer for this part
        out.release()

        metadata_out_path.write_text(json.dumps(param_dict, indent=4))

        # print(f"Saved video part: {video_out_path}")

    # Release the video capture object
    cap.release()


def main():
    video_root_path = Path('D:/')
    out_path = video_root_path / 'per_signal_videos'
    out_path.mkdir(parents=True, exist_ok=True)
    interation_data = match_videos_to_signals()
    for video_path, param_dicts in interation_data.items():
        save_video_parts(video_path, out_path, param_dicts)


if __name__ == '__main__':
    main()
