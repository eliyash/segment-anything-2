import pickle
import time

import cv2
from pathlib import Path

import numpy as np


def main():
    print(f'Global start')

    frames_folder = Path(r"D:/per_signal_videos_fixed_interlacing")
    dataset_folder = Path(r"D:/frames_collection_per_signal_fixed_interlacing")

    frames_skip = 300
    dataset_folder.mkdir(exist_ok=True, parents=True)
    video_files_only = [path for path in frames_folder.iterdir() if path.suffix != '.json']

    for video_path in video_files_only:
        print(f'\tStarting {video_path.name}')
        video_capture = cv2.VideoCapture(video_path.as_posix())

        if not video_capture.isOpened():
            print(f"Error opening video file: {video_path.name}")
            continue

        # get video frames count
        frames_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frames_count < frames_skip:
            amount_of_frames = 1
        else:
            amount_of_frames = frames_count // frames_skip

        interval = frames_count // (amount_of_frames + 1)

        frames_indices = [interval * (1+i) for i in range(amount_of_frames)]

        for frame_index in frames_indices:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video_capture.read()
            assert ret

            frame_path = dataset_folder / f"{video_path.stem}__frame_{frame_index}.png"
            cv2.imwrite(frame_path.as_posix(), frame)
        video_capture.release()

        print(f'\tTotal: {frames_count}, frames_count: {frames_count}, indexes: {frames_indices}')
        print(f'\tFinished')


if __name__ == '__main__':
    main()
