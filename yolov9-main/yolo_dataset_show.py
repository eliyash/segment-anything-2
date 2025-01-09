from pathlib import Path
import cv2
from monkey_names_with_classes import ALL_CLASS_INDEX_TO_NAMES


class YoloDataset:
    def __init__(self, dataset_root_folder, dataset_type='train'):
        self.dataset_root_folder = Path(dataset_root_folder)
        self.dataset_type = dataset_type
        self.image_dir = self.dataset_root_folder / "images" / dataset_type
        self.label_dir = self.dataset_root_folder / "labels" / dataset_type
        self.image_files = sorted([f.name for f in self.image_dir.glob("*")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_dir / self.image_files[index]
        label_path = self.label_dir / f'{Path(self.image_files[index]).stem}.txt'

        image = cv2.imread(str(image_path))
        height, width, _ = image.shape
        bboxes = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)
                bboxes.append([class_id, x_min, y_min, x_max, y_max])

        return image, bboxes

    def get_faces(self, index, margine=0.2):
        image, bboxes = self[index]

        faces = {}
        for bbox in bboxes:
            class_id, x_min, y_min, x_max, y_max = map(int, bbox)
            # increase hight and width by margine
            x_min = max(0, x_min - int((x_max - x_min) * margine))
            y_min = max(0, y_min - int((y_max - y_min) * margine))
            x_max = min(image.shape[1], x_max + int((x_max - x_min) * margine))
            y_max = min(image.shape[0], y_max + int((y_max - y_min) * margine))

            face = image[y_min:y_max, x_min:x_max]
            real_name = ALL_CLASS_INDEX_TO_NAMES[class_id]
            faces[real_name] = face

        return faces

    def show_image(self, index):
        image, bboxes = self[index]

        for bbox in bboxes:
            class_id, x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_faces():
    dataset_root_folder = r'D:\count_crop_and_recognise_dataset\refactor'
    # dataset_root_folder = r'C:\Users\Eliahu\Downloads\coco_datasets\chimps'
    dataset = YoloDataset(dataset_root_folder, dataset_type='test')
    size_of_dataset = len(dataset)
    for i in range(0, size_of_dataset, size_of_dataset//20):
        try:
            dataset.show_image(index=i)
        except Exception as e:
            print(f"Error displaying image at index {i}: {e}")


def save_face_dataset():
    root = Path(r'D:/')
    output_folder = root / 'faces_dataset'
    dataset_name_to_folder = {
        'chimpid': (Path(r'C:\Workspace\ChimpanzeesThesis\chimpanzee_id_data_yolo_fmt'), 1),
        'ccr': (root / 'count_crop_and_recognise_dataset' / 'refactor', 15),
    }
    for dataset_name, (dataset_folder, interval) in dataset_name_to_folder.items():
        for dataset_type in ['train', 'val', 'test']:
            dataset = YoloDataset(dataset_folder, dataset_type=dataset_type)
            size_of_dataset = len(dataset)

            for image_index in range(0, size_of_dataset, interval):
                faces_dict = dataset.get_faces(index=image_index)
                for real_name, face in faces_dict.items():
                    file_path = output_folder / dataset_type / f'{dataset_name}_{real_name}' / dataset.image_files[image_index]
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    if not file_path.exists():
                        cv2.imwrite(str(file_path), face)


if __name__ == "__main__":
    show_faces()
    # save_face_dataset()
