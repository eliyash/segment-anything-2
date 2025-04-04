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
    for key, (label, bbox, *_) in objects.items():
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
    annotations_folder = root_folder / 'annotations_head_and_body'
    images_folder = root_folder / 'raw_images'
    ndjson_file = root_folder / 'Export  project - video frames of chimps in Los Angeles Zoo - 3_20_2025.ndjson'


    images_dict = {image_path.name: image_path for image_path in images_folder.glob('*.png')}

    all_images_data = {}
    with open(ndjson_file, 'r') as file:
        for line_index, line in enumerate(file):
            json_obj = json.loads(line)

            file_name = json_obj['data_row']['external_id']

            annotations = list(json_obj['projects'].values())[0]['labels'][0]['annotations']
            all_objects = annotations['objects']
            relationships = annotations['relationships']

            all_annotations_raw = {}
            for ind, annotation_object in enumerate(all_objects):
                if annotation_object['name'] == 'head':
                    all_annotations_raw[annotation_object['feature_id']] = (annotation_object['name'], annotation_object['bounding_box'], annotation_object['classifications'][0]['radio_answer']['value'])
                else:
                    all_annotations_raw[annotation_object['feature_id']] = (annotation_object['name'], annotation_object['bounding_box'])

            boxes_by_animal = {}

            map_of_faces_to_heads = get_map_of_faces_to_heads(all_annotations_raw)
            for i, relationship in enumerate(relationships):
                relationship_map = relationship['unidirectional_relationship']
                primate = all_annotations_raw[relationship_map['source']]
                head = all_annotations_raw[relationship_map['target']]
                # fix wrong arrow direction
                face = None
                if relationship_map['target'] in map_of_faces_to_heads:
                    face = all_annotations_raw[map_of_faces_to_heads[relationship_map['target']]]

                if head[0] != 'head' or primate[0] != 'primate' or (face is not None and face[0] != 'face'):
                    print(f'issue in line index: {line_index}, primate: {primate}, head: {head} face: {face}')
                else:
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


def fix_posture_name(posture) -> str:
    # {'partialy_ocluded/blurry': 1568, 'frontal': 711, 'blurry': 2, 'angeled': 541, 'profile': 807, 'back': 1176}
    if 'blurry' in posture:
        posture = 'occluded_or_blurry'
    elif 'angeled' == posture:
        posture = 'angled'
    return posture

def get_scaled_bbox(bbox, img_shape):
    imh, imw, _ = img_shape
    x, y, w, h = map(int, [bbox['left'], bbox['top'], bbox['width'], bbox['height']])
    x, y, w, h = x / imw, y / imh, w / imw, h / imh
    return (x + w / 2), (y + h / 2), w, h

def get_image_label(boxes_by_animal, img_shape, save_face_only=False):
    imh, imw, _ = img_shape
    list_of_bboxes = []
    for _, (primate_data, head_data, face_data) in boxes_by_animal.items():
        if save_face_only:
            # consider using unknown as before
            face_bbox = face_data[1]
            list_of_bboxes.append((ALL_NAMES_TO_CLASS_INDEX['CHIMP_FACE'], *get_scaled_bbox(face_bbox, img_shape)))
        else:
            head_bbox = head_data[1]
            posture_name = f'CHIMP_HEAD_{fix_posture_name(head_data[2]).upper()}'
            body_bbox = primate_data[1]
            list_of_bboxes.append((ALL_NAMES_TO_CLASS_INDEX[posture_name], *get_scaled_bbox(head_bbox, img_shape)))
            list_of_bboxes.append((ALL_NAMES_TO_CLASS_INDEX['CHIMP_BODY'], *get_scaled_bbox(body_bbox, img_shape)))

    return list_of_bboxes

def get_yolov9_image_label(list_of_bboxes) -> str:
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