import pickle
import time

import cv2
from pathlib import Path

import numpy as np


def main():
    print(f'after mask_generator {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # use opencv to iterate on frames of a video

    video_root_path = Path(r'D:\videos_2019')  # Replace with your video file path
    annotation_root_folder = Path(r"C:\Workspace\ChimpanzeesThesis\outputs\sam_2")
    for video_path in video_root_path.iterdir():
        video_annotation_folder = annotation_root_folder / video_path.stem
        if not video_annotation_folder.exists():
            continue

        # Open the video file
        video_capture = cv2.VideoCapture(video_path.as_posix())  # Replace with your video file path

        # Check if the video file was opened successfully
        if not video_capture.isOpened():
            print("Error opening video file")

        # Loop through the video frames
        i = 0
        while True:
            # Read the next frame
            ret, frame = video_capture.read()
            # If there are no more frames, break the loop
            if not ret:
                break
            file_path = video_annotation_folder / f'masks_{i}.pkl'
            i += 1
            if not file_path.exists():
                break
            #     read pickle file
            with open(file_path, 'rb') as f:
                masks = pickle.load(f)
            # for evry mask, shift image values of the mask to a random color, e.g. mask one more redish, 2 blueish etc.
            for mask in masks:
                seg = mask['segmentation']
                frame[seg] += (20 * np.random.random(3)).astype('uint8')
            cv2.imshow('image', frame)
            cv2.waitKey(1)



if __name__ == '__main__':
    main()
