import json
import shutil
from pathlib import Path

import cv2
import parse

JUST_HORIZONTAL_LOCATIONS = ['l', 'm', 'r']
JUST_VERTICAL_LOCATIONS = ['t', 'm', 'b']

import random
import shutil
from pathlib import Path

def split_and_save_data(all_cases, dataset_folder_path):
    """
    Shuffles and splits data into train, validation, and test sets, then saves it to the specified folder.

    Args:
        all_cases: A list of tuples, where each tuple is (label_data, image_file_path).
        dataset_folder_path: The path to the folder where the data should be saved.
    """

    dataset_folder_path = Path(dataset_folder_path)

    # Create necessary directories
    (dataset_folder_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_folder_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_folder_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_folder_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_folder_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_folder_path / "labels" / "test").mkdir(parents=True, exist_ok=True)

    # Shuffle the data
    random.shuffle(all_cases)

    # Split the data
    train_split = int(0.8 * len(all_cases))
    val_split = int(0.95 * len(all_cases))
    train_cases = all_cases[:train_split]
    val_cases = all_cases[train_split:val_split]
    test_cases = all_cases[val_split:]

    # Process each dataset type
    for dataset_type, cases in zip(["train", "val", "test"], [train_cases, val_cases, test_cases]):
        # Open the file to write image paths
        with open(dataset_folder_path / f"{dataset_type}.txt", "w") as image_list_file:
            for label_data, image_file_path in cases:
                image_file_path = Path(image_file_path)
                image_name = image_file_path.name
                image_stem = image_file_path.stem

                # Copy the image
                destination_image_path = dataset_folder_path / "images" / dataset_type / image_name
                shutil.copy(image_file_path, destination_image_path)

                # Write the label data
                label_file_path = dataset_folder_path / "labels" / dataset_type / f"{image_stem}.txt"
                with open(label_file_path, "w") as label_file:
                    for inner_list in label_data:
                        label_file.write(" ".join(map(str, inner_list)) + "\n")

                # Write the relative image path
                relative_image_path = f"./images/{dataset_type}/{image_name}"
                image_list_file.write(relative_image_path + "\n")


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
    # Open the NDJSON file and read line by line

    all_cases = []
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
            # print(boxes_by_animal)

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
            all_cases.append((label_data, image_file_path))

    save_folder = Path(r'C:\Users\Eliahu\Downloads\coco_datasets\chimps')
    save_folder.mkdir(exist_ok=True, parents=True)
    split_and_save_data(all_cases, save_folder)


def get_image_label(boxes_by_animal, img_shape):
    imh, imw, _ = img_shape
    list_of_bboxes = []
    for individual_name, data in boxes_by_animal.items():
        face_bbox = [box for part_name, box in data if part_name == 'face'][0]
        x, y, w, h = map(int, [face_bbox['left'], face_bbox['top'], face_bbox['width'], face_bbox['height']])
        class_ind = 0  # use individual name for class
        list_of_bboxes.append((class_ind, (x + w / 2) / imw, (y + h / 2) / imh, w / imw, h / imh))
    return list_of_bboxes


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