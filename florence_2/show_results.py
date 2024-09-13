import json
from pathlib import Path

import cv2


def monkey_show(image, bboxs, colors=None):
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 120, 0), (0, 125, 120), (120, 0, 125), (85, 85, 85)]

    image_with_boxes = image.copy()

    for i, (l, t, r, b) in enumerate(bboxs['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']):
        (l, t, r, b) = map(int, (l, t, r, b))
        cv2.rectangle(image_with_boxes, (l, t), (r, b), colors[i % len(colors)], 2)

    return image_with_boxes


def show_florence2_results(video_root_path, annotation_root):
    for video_path in video_root_path.iterdir():

        body_annotation_path = annotation_root / video_path.stem
        face_annotation_path = annotation_root / 'face' / video_path.stem

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
            body_file_path = body_annotation_path / f'bboxs_{i}.json'
            face_file_path = face_annotation_path / f'bboxs_{i}.json'
            if not body_file_path.exists() or not face_file_path.exists():
                break
            body_res_dict = json.loads(body_file_path.read_text())
            face_res_dict = json.loads(face_file_path.read_text())
            frame_with_bboxs = monkey_show(frame, body_res_dict, [(255, 0, 0)])
            frame_with_bboxs = monkey_show(frame_with_bboxs, face_res_dict, [(0, 0, 255)])
            cv2.imshow('image', frame_with_bboxs)
            cv2.waitKey(100)
            i += 1


def main():
    video_root_path = Path(r'D:\videos_2019')
    annotation_root = Path(r'C:\Workspace\ChimpanzeesThesis\outputs\florence2_current\florence2')

    show_florence2_results(video_root_path, annotation_root)


if __name__ == '__main__':
    main()
