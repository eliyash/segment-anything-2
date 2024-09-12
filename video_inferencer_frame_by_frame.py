import json
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
    np.random.seed(3)
    device = 'cuda'
    # device = 'cpu'
    sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
    print(f'before build_sam2 {time.strftime("%Y-%m-%d %H:%M:%S")}')
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    print(f'after build_sam2 {time.strftime("%Y-%m-%d %H:%M:%S")}')
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    print(f'after mask_generator {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # use opencv to iterate on frames of a video

    video_path = Path('your_video_file.mp4')  # Replace with your video file path

    output_folder = Path("output") / video_path.stem
    output_folder.mkdir(exist_ok=True, parents=True)

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
        filename = output_folder / f'masks_{i}.png'
        masks = mask_generator.generate(frame)
        filename.write_text(json.dumps(masks))
        i += 1


if __name__ == '__main__':
    main()
