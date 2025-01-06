import json
from pathlib import Path

import cv2
import numpy as np

from create_yolov9_ccr_dataset__23_12_2024 import save_split_and_save_dataset_in_yolo_format

TYPE_OF_DATA_BY_INDEX = [
    'Nose', 'Left eye', 'Right eye', 'Head', 'Neck', 'Left shoulder', 'Left elbow', 'Left wrist', 'Right shoulder',
    'Right elbow', 'Right wrist', 'Hip/Sacrum', 'Left knee', 'Left foot', 'Right knee', 'Right foot'
]

HEAD_FEATURES = ['Nose', 'Left eye', 'Right eye', 'Head', 'Neck']
HEAD_FEATURES_INDEXES = [TYPE_OF_DATA_BY_INDEX.index(feature) for feature in HEAD_FEATURES]
UNKNOWN_CHIMP_IDENTITY__CLASS = 18


def get_eclipse_enclosing_bbox(center_x, center_y, major_axis, minor_axis, angle, scale=1.0):
    theta_radians = np.radians(angle)
    cos_theta = np.cos(theta_radians)
    sin_theta = np.sin(theta_radians)

    sqrt_x = np.sqrt(major_axis**2 * cos_theta**2 + minor_axis**2 * sin_theta**2) * scale
    sqrt_y = np.sqrt(major_axis**2 * sin_theta**2 + minor_axis**2 * cos_theta**2) * scale

    min_xy = int(center_x - sqrt_x), int(center_y - sqrt_y)
    max_xy = int(center_x + sqrt_x), int(center_y + sqrt_y)
    return min_xy, max_xy


def draw_face_ellipse(nose, right_eye, left_eye, head_top, neck):
    eye1_x, eye1_y = right_eye
    eye2_x, eye2_y = left_eye
    nose_x, nose_y = nose

    # Calculate center (nose as center)
    center_x = nose_x
    center_y = nose_y

    # Calculate angle
    eye_midpoint_x = (eye1_x + eye2_x) / 2
    eye_midpoint_y = (eye1_y + eye2_y) / 2
    angle = np.degrees(np.arctan2(nose_y - eye_midpoint_y, nose_x - eye_midpoint_x))

    head_neck_dist = np.sqrt((neck[0] - head_top[0])**2 + (neck[1] - head_top[1])**2)

    # Calculate axes lengths
    major_axis = int(np.sqrt((eye1_x - eye2_x) ** 2 + (eye1_y - eye2_y) ** 2) * 1.5)
    if major_axis < head_neck_dist * 0.5:
        major_axis = int(head_neck_dist * 0.5)
    minor_axis = int(major_axis * 0.7)

    # # --- Draw head ellipse ---
    # cv2.ellipse(image, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, (0, 255, 0), 2)

    min_xy, max_xy = get_eclipse_enclosing_bbox(center_x, center_y, major_axis, minor_axis, angle, 1.4)
    return min_xy, max_xy


def main():
    root_folder = Path(r'D:\open_ape_pose')
    all_annotations_per_image = {}

    all_images_folder = root_folder / 'all_images'
    annotations_file = root_folder / 'annotations' / 'oap_all.json'
    raw_annotations = json.loads(annotations_file.read_text())
    annotations = raw_annotations['data']
    for sample_data in annotations:
        if sample_data['species'] != 'Chimpanzee':
            # print(sample_data['species'])
            continue
        landmarks = np.array(sample_data['landmarks'])
        landmarks = landmarks.reshape(-1, 2)

        image_path = all_images_folder / sample_data['file']
        # image = cv2.imread(str(image_path))
        # bbox = sample_data['bbox']
        # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 125, 125), 2)

        valid_features = []
        for feature_id, (feature_data, is_valid) in enumerate(zip(landmarks, sample_data['visibility'])):
            if feature_id in HEAD_FEATURES_INDEXES and is_valid:
                valid_features.append(feature_data)
                # x, y = feature_data
                # cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        if len(valid_features) == len(HEAD_FEATURES_INDEXES):
            valid_features = np.array(valid_features)

            image_h, image_w, _ = cv2.imread(str(image_path)).shape
            image_shape = np.array([image_w, image_h])

            min_xy, max_xy = draw_face_ellipse(*valid_features)
            min_xy, max_xy = np.array(min_xy), np.array(max_xy)
            min_xy, max_xy = min_xy / image_shape, max_xy / image_shape

            w, h = max_xy - min_xy
            x_center, y_center = (max_xy + min_xy) / 2
            chimp_line = f'{UNKNOWN_CHIMP_IDENTITY__CLASS} {x_center} {y_center} {w} {h}\n'
            relative_path = image_path.relative_to(all_images_folder)
            assert relative_path not in all_annotations_per_image
            all_annotations_per_image[relative_path] = chimp_line

    save_split_and_save_dataset_in_yolo_format(
        root_folder / 'yolo_fmt', all_images_folder, all_annotations_per_image, 640
    )


if __name__ == '__main__':
    main()
