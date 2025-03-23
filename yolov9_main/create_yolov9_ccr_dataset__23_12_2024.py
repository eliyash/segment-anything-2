import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from monkey_names_with_classes import ALL_NAMES_TO_CLASS_INDEX, IGNORE_NAME


def main():
    root_folder = Path(r'D:\count_crop_and_recognise_dataset')
    original_images_folder = root_folder / 'images'
    all_frames_data = {}
    for year in [2012, 2013]:
        for movie_path in (original_images_folder / str(year)).iterdir():
            image_paths = list(movie_path.glob('*.jpg'))
            im_h, im_w, _ = cv2.imread(str(image_paths[0])).shape
            fix_h = im_h / im_w

            for image_path in image_paths:
                annotation_path = movie_path / f'face_{image_path.stem}.json'
                if annotation_path.exists():
                    face_annotations = json.loads(annotation_path.read_text())
                    fixed_face_annotations = {}
                    for individual_name, (x, y, w, h) in face_annotations.items():
                        fixed_face_annotations[individual_name.lower()] = (x, y * fix_h, w, h)

                    try:
                        yolo_annotation = get_yolov9_image_label(fixed_face_annotations)
                        all_frames_data[image_path.relative_to(original_images_folder)] = yolo_annotation
                    except ValueError:
                        print(f"No bounding boxes found in {image_path}")

    save_split_and_save_dataset_in_yolo_format(root_folder / 'refactor', original_images_folder, all_frames_data)


def save_split_and_save_dataset_in_yolo_format(new_dataset_folder, original_images_folder, all_frames_data, max_height=None):
    all_image_files = np.array(list(all_frames_data.keys()))
    np.random.seed(42)
    np.random.shuffle(all_image_files)
    train_split = int(0.8 * len(all_image_files))
    val_split = int(0.95 * len(all_image_files))

    train_cases = all_image_files[:train_split]
    val_cases = all_image_files[train_split:val_split]
    test_cases = all_image_files[val_split:]

    new_dataset_images_folder = new_dataset_folder / 'images'
    new_dataset_labels_folder = new_dataset_folder / 'labels'
    for dataset_type, cases in zip(["train", "val", "test"], [train_cases, val_cases, test_cases]):
        new_image_paths = []
        for relative_orig_image_path in cases:
            yolo_annotation = all_frames_data[relative_orig_image_path]
            # new_image_file_name = '_'.join(relative_orig_image_path.parts)
            new_image_file_name = relative_orig_image_path
            label_path = new_dataset_labels_folder / dataset_type / f'{Path(new_image_file_name).stem}.txt'
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(yolo_annotation)

            image_path = new_dataset_images_folder / dataset_type / new_image_file_name
            if not image_path.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if max_height is not None:
                    image = cv2.imread(str(original_images_folder / relative_orig_image_path))
                    image_h, image_w, _ = image.shape
                    if image_h > max_height:
                        width_scale = image_w / image_h
                        image = cv2.resize(image, (int(max_height * width_scale), max_height))
                    cv2.imwrite(str(image_path), image)
                else:
                    shutil.copy(str(original_images_folder / relative_orig_image_path), str(image_path))
                new_image_paths.append(image_path.relative_to(new_dataset_folder))

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
