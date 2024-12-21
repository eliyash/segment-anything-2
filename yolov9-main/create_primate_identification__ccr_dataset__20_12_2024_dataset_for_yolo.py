import json
import shutil
from pathlib import Path

import cv2
from create_primate_identification__8_18_2024_dataset_for_yolo import split_and_save_data

IGNORE_NAME = 'NEGATIVE'
NAMES = ['TUA', 'PELEY', 'PAMA', 'VELU', 'JEJE', 'JIRE', 'FANA', 'FLANLE', 'FOAF', 'FANWA', 'FANLE', 'JOYA', 'YO']
NAME_TO_CLASS_INDEX = {name: i+25 for i, name in enumerate(NAMES)}


def main():
    root_folder = Path(r'D:\count_crop_and_recognise_dataset\frame_cv')
    all_cases = []
    for image_path in root_folder.rglob('*/*/*.jpg'):
        movie_path = image_path.parent
        frame_number = image_path.stem
        annotations = {}
        for type_of_data in ['body', 'face']:
            annotation_path = movie_path / f'{type_of_data}_{frame_number}.json'
            if annotation_path.exists():
                annotations[type_of_data] = json.loads(annotation_path.read_text())
        if 'face' in annotations:
            label_data = get_image_label(annotations['face'])
            all_cases.append((label_data, image_path))

    save_folder = Path(r'D:\count_crop_and_recognise_dataset\yolov9_style')
    save_folder.mkdir(exist_ok=True, parents=True)
    (save_folder / 'name_to_class_index.json').write_text(json.dumps(NAME_TO_CLASS_INDEX, indent=4))
    split_and_save_data(all_cases, save_folder)


def get_image_label(boxes_by_animal):
    list_of_bboxes = []
    for individual_name, (x, y, w, h) in boxes_by_animal.items():
        if individual_name != IGNORE_NAME:
            list_of_bboxes.append((NAME_TO_CLASS_INDEX[individual_name], (x + w / 2), (y + h / 2), w, h))
    return list_of_bboxes


if __name__ == '__main__':
    main()
