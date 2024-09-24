# import json
# import pickle
# import time
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# import torch
# from sam2.sam2_video_predictor import SAM2VideoPredictor
#
#
# def main():
#     np.random.seed(3)
#     device = 'cuda'
#     print(f'before build_sam2 {time.strftime("%Y-%m-%d %H:%M:%S")}')
#
#     predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2_hiera_tiny")
#
#     video_path = Path('/home/ubuntu/videos_2019/6_20_19 (12).MP4')  # Replace with your video file path
#     with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
#         state = predictor.init_state(video_path)
#
#         # add new prompts and instantly get the output on the same frame
#         frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, < your_prompts >):
#
#         # propagate the prompts to get masklets throughout the video
#         for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#             ...
#     # use opencv to iterate on frames of a video
#
#     video_path = Path('/home/ubuntu/videos_2019/6_20_19 (12).MP4')  # Replace with your video file path
#
#     output_folder = Path("output") / video_path.stem
#     output_folder.mkdir(exist_ok=True, parents=True)
#
#     # Open the video file
#     video_capture = cv2.VideoCapture(video_path.as_posix())  # Replace with your video file path
#
#     # Check if the video file was opened successfully
#     if not video_capture.isOpened():
#         print("Error opening video file")
#
#     # Loop through the video frames
#     i = 0
#     while True:
#         # Read the next frame
#         ret, frame = video_capture.read()
#         # If there are no more frames, break the loop
#         if not ret:
#             break
#         file_path = output_folder / f'masks_{i}.png'
#         masks = mask_generator.generate(frame)
#         with open(file_path, 'wb') as f:  # 'wb' for writing in binary mode
#             pickle.dump(masks, f)
#         i += 1
#
#
# if __name__ == '__main__':
#     main()
