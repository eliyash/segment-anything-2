import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

from yolov9_main.detection_and_tracking_of_full_videos import get_capture_frame


def main():
    data_name = 'per_signal_videos_fixed_interlacing'
    data_folder = Path('D:/')
    local_data_folder = Path('C:\Workspace\ChimpanzeesThesis\data_local_copy')
    if (local_data_folder / data_name).exists():
        source = local_data_folder / data_name
    else:
        source = data_folder / data_name

    video_paths = sorted(
        [file_path for file_path in source.glob('*') if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']]
    )

    for video_path in video_paths:
        inference_video(video_path)


def inference_video(video_path):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), video_path)
    out_folder = Path(r'C:\Workspace\ChimpanzeesThesis\signal_video_output\faces_tracking')
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Video Not Found {video_path}"

    accumulated_tracked_objects = json.loads((out_folder / f'accumulated_tracked_objects_{video_path.stem}.json').read_text())
    per_frame_data = defaultdict(dict)
    for uid, bboxes_by_frame_ind in accumulated_tracked_objects.items():
        for frame_ind, bbox in bboxes_by_frame_ind.items():
            per_frame_data[int(frame_ind)][uid] = bbox

    frame_index = -1
    while True:
        frame = get_capture_frame(cap)
        if frame is None:
            break

        if frame_index in per_frame_data:
            detected_bboxes = per_frame_data[frame_index]
            for uid, (_, (x_start, x_end), (y_start, y_end)) in detected_bboxes.items():
                # uid_folder = out_folder / uid
                # uid_folder.mkdir(exist_ok=True)
                # cv2.imwrite(str(uid_folder / f'{frame_index}.png'), crop_with_padding_from_bounds(frame, x_start, x_end, y_start, y_end))
                cv2.rectangle(frame, (y_start, x_start), (y_end, x_end), (0, 255, 0), 2)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        frame_index += 1
    cap.release()


if __name__ == "__main__":
    main()