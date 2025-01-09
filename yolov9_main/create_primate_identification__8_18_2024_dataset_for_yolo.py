import json

import cv2
import parse

from pathlib import Path

from create_yolov9_ccr_dataset__23_12_2024 import save_split_and_save_dataset_in_yolo_format, get_yolov9_image_label, \
    ALL_NAMES_TO_CLASS_INDEX

JUST_HORIZONTAL_LOCATIONS = ['l', 'm', 'r']
JUST_VERTICAL_LOCATIONS = ['t', 'm', 'b']


def get_sorted_names_by_location(names_by_locations, is_horizontal):
    reference_list = JUST_HORIZONTAL_LOCATIONS if is_horizontal else JUST_VERTICAL_LOCATIONS
    sorted_names = sorted(names_by_locations.keys(), key=lambda n: reference_list.index(n))
    return [names_by_locations[name] for name in sorted_names]


def get_sorted_indexes_by_location(boxes_by_animal, is_horizontal):
    map_center_of_axis_to_data = {}
    for ind, data in boxes_by_animal.items():
        face_bbox = [box for part_name, box in data if part_name == 'face'][0]
        x, y, w, h = map(int, [face_bbox['left'], face_bbox['top'], face_bbox['width'], face_bbox['height']])
        center = x + w // 2 if is_horizontal else y + h // 2
        map_center_of_axis_to_data[center] = data
    sorted_centers = sorted(map_center_of_axis_to_data.keys())
    return [map_center_of_axis_to_data[center] for center in sorted_centers]


def main():
    root_folder = Path(r'C:\Workspace\ChimpanzeesThesis')
    project_folder = root_folder / 'Chimpanzee ID Data'
    all_names_lower = [f.name.lower() for f in project_folder.iterdir() if f.is_dir()]
    images_dict = {}
    for image_path in project_folder.glob('**/*.jpg'):
        images_dict[image_path.name] = image_path

    all_images_data = {}
    duplicates_cases = []
    complicated_relations_cases = []
    with open(project_folder / 'Export v2 project - primate identification - 8_18_2024.ndjson', 'r') as file:
        for line in file:
            json_obj = json.loads(line)

            file_name = json_obj['data_row']['external_id']
            # print(file_name, images_dict[file_name])

            annotations = list(json_obj['projects'].values())[0]['labels'][0]['annotations']
            all_objects = annotations['objects']
            relationships = annotations['relationships']

            all_annotations_raw = {}
            for ind, annotation_object in enumerate(all_objects):
                all_annotations_raw[annotation_object['feature_id']] = (annotation_object['name'], annotation_object['bounding_box'])

            boxes_by_animal = {}
            for i, relationship in enumerate(relationships):
                relationship_map = relationship['unidirectional_relationship']
                boxes_by_animal[i] = all_annotations_raw[relationship_map['source']], all_annotations_raw[relationship_map['target']]

            file_name_lower = file_name.lower()
            names_in_image = list(filter(lambda n: n in file_name_lower, all_names_lower))
            if len(boxes_by_animal) != len(names_in_image):
                duplicates_cases.append(file_name)
                continue

            if len(boxes_by_animal) == 1:
                bboxs_by_names = {names_in_image[0]: boxes_by_animal[0]}

            elif len(boxes_by_animal) >= 2:
                matches = {match.named['loc']: match.named['name'] for match in parse.findall('{name}({loc})', file_name_lower.replace('_', '')) if match.named['name'] in all_names_lower}
                locations = set(matches.keys())
                is_horizontal_orientation = locations.issubset(JUST_HORIZONTAL_LOCATIONS)
                is_vertical_orientation = locations.issubset(JUST_VERTICAL_LOCATIONS)
                if not is_horizontal_orientation and not is_vertical_orientation:
                    complicated_relations_cases.append(file_name)
                    continue

                order_names_by_location = get_sorted_names_by_location(matches, is_horizontal_orientation)
                ordered_indexes_by_location = get_sorted_indexes_by_location(boxes_by_animal, is_horizontal_orientation)

                bboxs_by_names = {name: bboxs for name, bboxs in zip(order_names_by_location, ordered_indexes_by_location)}

                #     print(names_in_image, file_name_lower, matches)

            image_file_path = images_dict[file_name]
            img_shape = cv2.imread(image_file_path.as_posix()).shape
            label_data = get_image_label(bboxs_by_names, img_shape)
            yolo_annotation = get_yolov9_image_label(label_data)
            all_images_data[image_file_path.relative_to(project_folder)] = yolo_annotation

    save_folder = Path(r'C:\Users\Eliahu\Downloads\coco_datasets\chimps_new')
    save_folder.mkdir(exist_ok=True, parents=True)
    save_split_and_save_dataset_in_yolo_format(root_folder / 'refactor', project_folder, all_images_data)
    # split_and_save_data(all_cases, save_folder)


def get_image_label(boxes_by_animal, img_shape):
    imh, imw, _ = img_shape
    scaled_bbox = {}
    for individual_name, data in boxes_by_animal.items():
        assert individual_name in ALL_NAMES_TO_CLASS_INDEX
        face_bbox = [box for part_name, box in data if part_name == 'face'][0]
        x, y, w, h = map(int, [face_bbox['left'], face_bbox['top'], face_bbox['width'], face_bbox['height']])
        scaled_bbox[individual_name] = x / imw, y / imh, w / imw, h / imh
    return scaled_bbox


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