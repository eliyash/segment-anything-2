import json
from pathlib import Path

import cv2

ALL_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 120, 0), (0, 125, 120), (120, 0, 125), (85, 85, 85)]
ALL_BODY_PARTS = ['chimpanzee', 'face', 'ear']
COLORS_DICT = {part: ALL_COLORS[color_ind] for color_ind, part in enumerate(ALL_BODY_PARTS)}


def look_for_match(holding_bbox, bboxs_to_check):
    (hl, ht, hr, hb) = holding_bbox
    hw = hr - hl
    hh = hb - ht
    (el, et, er, eb) = (hl - hw // 4, ht - hh // 4, hr + hw // 4, hb + hh // 4)
    matches = []
    for bbox_to_check in bboxs_to_check:
        (cl, ct, cr, cb) = bbox_to_check
        cw = cr - cl
        ch = cb - ct
        if cw < hw / 2 and ch < hh / 2 and el <= cl and et <= ct and er >= cr and eb >= cb:
            matches.append(bbox_to_check)
    return matches


def monkey_show_filtered(image, mapped_parts_for_each_chimpanzee: list):
    image_with_boxes = image.copy()

    for i, individual_parts in enumerate(mapped_parts_for_each_chimpanzee):
        for type_of_part, bboxs in individual_parts.items():
            for bbox in bboxs:
                (l, t, r, b) = map(int, bbox)
                cv2.rectangle(image_with_boxes, (l, t), (r, b), COLORS_DICT[type_of_part], 2)

    return image_with_boxes


def validate_data(dict_of_all_bbox_dicts):
    # check if bbox of face at least 2 times smaller than chimpanzee (full body) in each axis
    # check if bbox of ear at least 2 times smaller than bbox of face in each axis
    # stages: frst we will match each full body to face and ear checking face and ears bbox are mostly inside full body bbox
    # after we got match for each chimpanzee parts, we will check soze validity
    full_body_bboxs = dict_of_all_bbox_dicts['chimpanzee']['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    face_bboxs = dict_of_all_bbox_dicts['face']['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    ear_bboxs = dict_of_all_bbox_dicts['ear']['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    mapped_parts_for_each_chimpanzee = []
    for full_body_bbox in full_body_bboxs:
        matched_face_bboxs = look_for_match(full_body_bbox, face_bboxs)
        matched_ear_bboxs = []
        for face_bbox in face_bboxs:
            matched_ear_bboxs.extend(look_for_match(face_bbox, ear_bboxs))
        mapped_parts_for_each_chimpanzee.append({'chimpanzee': [full_body_bbox], 'face': matched_face_bboxs, 'ear': matched_ear_bboxs})

    return mapped_parts_for_each_chimpanzee


def get_all_annotation_paths(annotation_root, current_frame_ind, video_path):
    all_annotation_file_paths = {}
    for annotation_type in annotation_root.iterdir():
        if annotation_type.name not in COLORS_DICT:
            continue
        annotation_path = annotation_type / video_path.stem
        annotation_file_path = annotation_path / f'bboxs_{current_frame_ind}.json'
        all_annotation_file_paths[annotation_type.name] = annotation_file_path
    return all_annotation_file_paths


def show_florence2_results(video_root_path, annotation_root):
    for video_path in video_root_path.iterdir():
        # video_path = video_root_path / '5_21_19 (1).MTS'
        time_in_seconds = 30

        # Open the video file
        video_capture = cv2.VideoCapture(video_path.as_posix())  # Replace with your video file path

        # Check if the video file was opened successfully
        if not video_capture.isOpened():
            print("Error opening video file")

        # Get the frames per second (FPS) of the video
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Calculate the frame number corresponding to the time
        current_frame_ind = int(time_in_seconds * fps)

        # start from frame i
        # Set the frame number to start from
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_ind)

        while True:
            # Read the next frame
            ret, frame = video_capture.read()
            # If there are no more frames, break the loop
            if not ret:
                break

            all_annotation_file_paths = get_all_annotation_paths(annotation_root, current_frame_ind, video_path)

            current_frame_ind += 1

            if not all([f.exists() for f in all_annotation_file_paths.values()]):
                break
            res_dicts = {n: json.loads(f.read_text()) for n, f in all_annotation_file_paths.items()}

            list_of_data = validate_data(res_dicts)
            frame_with_bboxs = monkey_show_filtered(frame, list_of_data)
            cv2.imshow('image', frame_with_bboxs)
            cv2.waitKey(10)


def main():
    drive_path = Path('D:/') if Path('D:/').exists() else Path('E:/')
    video_root_path = drive_path / 'videos_2019'
    annotation_root = Path(r'C:\Workspace\ChimpanzeesThesis\outputs\florence2_17_9_24\home\ubuntu\segment-anything-2\florence_2\output\florence2')

    show_florence2_results(video_root_path, annotation_root)


if __name__ == '__main__':
    main()
