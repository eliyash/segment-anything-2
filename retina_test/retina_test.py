import time
from pathlib import Path

import cv2
from retinaface.pre_trained_models import get_model


def check_if_valid(annotation):
    return len(annotation) > 1 or annotation[0]['bbox'] != []


def add_boxs(image, annotation, color=(0, 255, 0)):
    new_image = image.copy()
    for face_data in annotation:
        x_start, y_start, x_end, y_end = map(int, face_data['bbox'])
        new_image = cv2.rectangle(new_image, (x_start, y_start), (x_end, y_end), color, 2)
    return new_image


def main():
    print(f'start {time.strftime("%Y-%m-%d %H:%M:%S")}')
    model = get_model("resnet50_2020-07-20", max_size=2048, device='cuda')
    model.eval()

    output_folder = Path(r"C:\Workspace\ChimpanzeesThesis\outputs\retinaface")
    frames_folder = Path(r"D:/frames_collection")

    # image_paths = [frames_folder / "BEN_JULIE___16_8_17 JERRARD (A)_2.eaf.png"]
    for image_path in frames_folder.iterdir():
        image = cv2.imread(image_path.as_posix())

        for confidence, color in {0.7: (255, 0, 0), 0.5: (0, 255, 0), 0.3: (0, 0, 255)}.items():
            annotation = model.predict_jsons(image, confidence_threshold=confidence)
            if check_if_valid(annotation):
                image = add_boxs(image, annotation, color)
                break

        # cv2.imwrite((output_folder / image_path.name).as_posix(), image)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    print(f'end {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
