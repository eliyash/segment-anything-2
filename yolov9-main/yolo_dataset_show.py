from pathlib import Path
import cv2


class YoloDataset:
    def __init__(self, dataset_root_folder, dataset_type='train'):
        """
        Initializes the YoloDataset object.

        Args:
          dataset_root_folder: Path to the parent directory containing "images" and "labels" folders.
          dataset_type: Type of the dataset (e.g., 'train', 'val', 'test').
                        Images are assumed to be under images_folder/dataset_type/images.
        """
        self.dataset_root_folder = Path(dataset_root_folder)
        self.dataset_type = dataset_type
        self.image_dir = self.dataset_root_folder / "images" / dataset_type
        self.label_dir = self.dataset_root_folder / "labels" / dataset_type
        self.image_files = sorted([f.name for f in self.image_dir.glob("*")])

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Loads and returns the image and its bounding boxes at the given index.

        Args:
          index: Index of the image to load.

        Returns:
          A tuple containing the image and a list of bounding boxes.
        """
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

    def show_image(self, index):
        """
        Displays the image with bounding boxes drawn on it.

        Args:
          index: Index of the image to display.
        """
        image, bboxes = self[index]

        for bbox in bboxes:
            class_id, x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


dataset_root_folder = r'D:\count_crop_and_recognise_dataset\refactor'
# dataset_root_folder = r'C:\Users\Eliahu\Downloads\coco_datasets\chimps'
dataset = YoloDataset(dataset_root_folder, dataset_type='test')
size_of_dataset = len(dataset)
for i in range(0, size_of_dataset, size_of_dataset//20):
    try:
        dataset.show_image(index=i)
    except Exception as e:
        print(f"Error displaying image at index {i}: {e}")
