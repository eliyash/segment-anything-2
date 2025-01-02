import json
import shutil
from pathlib import Path

import cv2
import numpy as np

TYPE_OF_DATA_BY_INDEX = [
    'Nose', 'Left eye', 'Right eye', 'Head', 'Neck', 'Left shoulder', 'Left elbow', 'Left wrist', 'Right shoulder',
    'Right elbow', 'Right wrist', 'Hip/Sacrum', 'Left knee', 'Left foot', 'Right knee', 'Right foot'
]

HEAD_FEATURES = ['Nose', 'Left eye', 'Right eye', 'Head', 'Neck']
HEAD_FEATURES_INDEXES = [TYPE_OF_DATA_BY_INDEX.index(feature) for feature in HEAD_FEATURES]


def main():
    root_folder = Path(r'D:\open_ape_pose')

    images_folder = root_folder / 'images'
    annotations_file = root_folder / 'annotations' / 'oap_all.json'
    raw_annotations = json.loads(annotations_file.read_text())
    annotations = raw_annotations['data']
    for sample_data in annotations:
        if sample_data['species'] != 'Chimpanzee':
            print(sample_data['species'])
            continue
        landmarks = np.array(sample_data['landmarks'])
        landmarks = landmarks.reshape(-1, 2)

        image_path = images_folder / sample_data['file']
        image = cv2.imread(str(image_path))
        bbox = sample_data['bbox']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        valid_features = []
        for feature_id, (feature_data, is_valid) in enumerate(zip(landmarks, sample_data['visibility'])):
            if feature_id in HEAD_FEATURES_INDEXES and is_valid:
                valid_features.append(feature_data)
                x, y = feature_data
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        if len(valid_features):
            valid_features = np.array(valid_features)
            xs = valid_features[:, 0]
            ys = valid_features[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # now scale the bbox of valid_features by 3 vertically and 2 horizontally
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            w = (x_max - x_min) * 3
            h = int((y_max - y_min) * 2)
            cv2.rectangle(image, (x_center - w // 2, y_center - h // 2), (x_center + w // 2, y_center + h // 2), (255, 0, 0), 2)

        cv2.imshow('image', image)
        cv2.waitKey(0)
    #
    #
    # all_frames_data = {}
    # for year in [2012, 2013]:
    #     for movie_path in (original_images_folder / str(year)).iterdir():
    #         image_paths = list(movie_path.glob('*.jpg'))
    #         im_h, im_w, _ = cv2.imread(str(image_paths[0])).shape
    #         fix_h = im_h / im_w
    #
    #         for image_path in image_paths:
    #             annotation_path = movie_path / f'face_{image_path.stem}.json'
    #             if annotation_path.exists():
    #                 face_annotations = json.loads(annotation_path.read_text())
    #                 fixed_face_annotations = {}
    #                 for individual_name, (x, y, w, h) in face_annotations.items():
    #                     fixed_face_annotations[individual_name.lower()] = (x, y * fix_h, w, h)
    #
    #                 try:
    #                     yolo_annotation = get_yolov9_image_label(fixed_face_annotations)
    #                     all_frames_data[image_path.relative_to(original_images_folder)] = yolo_annotation
    #                 except ValueError:
    #                     print(f"No bounding boxes found in {image_path}")
    #
    # save_split_and_save_dataset_in_yolo_format(root_folder, original_images_folder, all_frames_data)


def save_split_and_save_dataset_in_yolo_format(root_folder, original_images_folder, all_frames_data):
    all_image_files = np.array(list(all_frames_data.keys()))
    np.random.seed(42)
    np.random.shuffle(all_image_files)
    train_split = int(0.8 * len(all_image_files))
    val_split = int(0.95 * len(all_image_files))

    train_cases = all_image_files[:train_split]
    val_cases = all_image_files[train_split:val_split]
    test_cases = all_image_files[val_split:]

    new_dataset_folder = root_folder / 'refactor'
    new_dataset_images_folder = new_dataset_folder / 'images'
    new_dataset_labels_folder = new_dataset_folder / 'labels'
    for dataset_type, cases in zip(["train", "val", "test"], [train_cases, val_cases, test_cases]):
        new_image_paths = []
        for relative_orig_image_path in cases:
            yolo_annotation = all_frames_data[relative_orig_image_path]
            new_image_file_name = '_'.join(relative_orig_image_path.parts)
            label_path = new_dataset_labels_folder / dataset_type / f'{Path(new_image_file_name).stem}.txt'
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(yolo_annotation)

            image_path = new_dataset_images_folder / dataset_type / new_image_file_name
            if not image_path.exists():
                new_image_paths.append(image_path.relative_to(new_dataset_folder))
                image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(original_images_folder / relative_orig_image_path), str(image_path))

        posix_relative_paths = [f'./{path.as_posix()}' for path in new_image_paths]
        (new_dataset_folder / f"{dataset_type}.txt").write_text("\n".join(posix_relative_paths) + "\n")

    (new_dataset_folder / 'name_to_class_index.json').write_text(json.dumps(ALL_NAMES_TO_CLASS_INDEX, indent=4))


def get_yolov9_image_label(boxes_by_animal) -> str:
    list_of_bboxes = []
    for individual_name, (x, y, w, h) in boxes_by_animal.items():
        if individual_name != IGNORE_NAME:
            list_of_bboxes.append((ALL_NAMES_TO_CLASS_INDEX[individual_name], (x + w / 2), (y + h / 2), w, h))

    if len(list_of_bboxes) == 0:
        raise ValueError("No bounding boxes found in the image.")
    return "\n".join([" ".join(map(str, bbox)) for bbox in list_of_bboxes]) + "\n"


if __name__ == '__main__':
    main()
