import json
from typing import Dict, Tuple

import cv2
import numpy as np
import parse

from pathlib import Path

from create_yolov9_ccr_dataset__23_12_2024 import save_split_and_save_dataset_in_yolo_format, ALL_NAMES_TO_CLASS_INDEX
from yolov9_main.monkey_names_with_classes import UNKNOWN_NAME


def get_center(bbox: Dict[str, float]) -> np.ndarray:
    """Compute the center coordinates of a bounding box as a NumPy array."""
    return np.array([bbox['left'] + bbox['width'] / 2, bbox['top'] + bbox['height'] / 2])


def get_map_of_faces_to_heads(objects: Dict[str, Tuple[str, Dict[str, float]]]) -> Dict[str, str]:
    """Map each face to the closest head based on the center distance using NumPy."""
    heads = {}
    faces = {}

    # Separate heads and faces and compute their centers
    for key, (label, bbox) in objects.items():
        if label == 'head':
            heads[key] = get_center(bbox)
        elif label == 'face':
            faces[key] = get_center(bbox)

    # Map faces to the nearest head
    face_to_head_mapping = {}
    for face_key, face_center in faces.items():
        closest_head = min(heads, key=lambda head_key: np.linalg.norm(face_center - heads[head_key]))
        # face_to_head_mapping[face_key] = closest_head
        face_to_head_mapping[closest_head] = face_key

    return face_to_head_mapping

def main():
    root_folder = Path(r'D:\frames_collection_per_signal_fixed_interlacing_filtered')
    annotations_folder = root_folder / 'annotations'
    images_folder = root_folder / 'images'
    ndjson_file = root_folder / 'Export  project - video frames of chimps in Los Angeles Zoo - 3_20_2025.ndjson'


    images_dict = {image_path.name: image_path for image_path in images_folder.glob('*.png')}

    all_images_data = {}
    with open(ndjson_file, 'r') as file:
        for line in file:
            json_obj = json.loads(line)

            file_name = json_obj['data_row']['external_id']

            annotations = list(json_obj['projects'].values())[0]['labels'][0]['annotations']
            all_objects = annotations['objects']
            relationships = annotations['relationships']

            all_annotations_raw = {}
            for ind, annotation_object in enumerate(all_objects):
                all_annotations_raw[annotation_object['feature_id']] = (annotation_object['name'], annotation_object['bounding_box'])

            boxes_by_animal = {}

            map_of_faces_to_heads = get_map_of_faces_to_heads(all_annotations_raw)
            for i, relationship in enumerate(relationships):
                relationship_map = relationship['unidirectional_relationship']
                primate = all_annotations_raw[relationship_map['source']]
                head = all_annotations_raw[relationship_map['target']]
                if relationship_map['target'] in map_of_faces_to_heads:
                    face = all_annotations_raw[map_of_faces_to_heads[relationship_map['target']]]
                    boxes_by_animal[i] = (primate, head, face)

            if not len(boxes_by_animal):
                continue
            # all_images_data[file_name] = boxes_by_animal

            image_file_path = images_dict[file_name]
            img_shape = cv2.imread(image_file_path.as_posix()).shape
            label_data = get_image_label(boxes_by_animal, img_shape)
            yolo_annotation = get_yolov9_image_label(label_data)
            all_images_data[image_file_path.relative_to(images_folder)] = yolo_annotation

    # count_of_values_combined = sum([len(v) for v in all_images_data.values()])
    # save_folder = Path(r'C:\Users\Eliahu\Downloads\coco_datasets\chimps_new')
    # save_folder.mkdir(exist_ok=True, parents=True)
    save_split_and_save_dataset_in_yolo_format(annotations_folder, images_folder, all_images_data)
    # split_and_save_data(all_cases, save_folder)


def get_image_label(boxes_by_animal, img_shape):
    imh, imw, _ = img_shape
    scaled_bbox = {}
    for individual_name, data in boxes_by_animal.items():
        face_bbox = [box for part_name, box in data if part_name == 'face'][0]
        x, y, w, h = map(int, [face_bbox['left'], face_bbox['top'], face_bbox['width'], face_bbox['height']])
        scaled_bbox[individual_name] = x / imw, y / imh, w / imw, h / imh
    return scaled_bbox

def get_yolov9_image_label(boxes_by_animal) -> str:
    list_of_bboxes = []
    for (x, y, w, h) in boxes_by_animal.values():
        list_of_bboxes.append((ALL_NAMES_TO_CLASS_INDEX[UNKNOWN_NAME], (x + w / 2), (y + h / 2), w, h))

    if len(list_of_bboxes) == 0:
        raise ValueError("No bounding boxes found in the image.")
    return "\n".join([" ".join(map(str, bbox)) for bbox in list_of_bboxes]) + "\n"

def show_image(boxes_by_animal, image_file_path):
    image = cv2.imread(str(image_file_path))
    number_of_monkeys = 1 + len(boxes_by_animal)
    cv2.imshow('image', image)
    for ind, data in boxes_by_animal.items():
        for part_name, box in data:
            x, y, w, h = map(int, [box['left'], box['top'], box['width'], box['height']])
            cv2.rectangle(image, (x, y), (x + w, y + h),
                          (0, 255 * (1 - ind / (number_of_monkeys - 1)), 255 * (ind / (number_of_monkeys - 1))), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()