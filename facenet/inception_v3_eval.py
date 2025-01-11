import argparse

import numpy as np
import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

from facenet.inception_v3_train import MODEL_NAME, ALL_CLASS_INDEX_TO_NAMES, ChimpFaceDataset


def main(config):
    phase_name = 'train'
    batch_size = config.batch_size
    output_path = Path(config.output_path)
    phase_data_path = Path(config.data_path) / phase_name
    output_path.mkdir(exist_ok=True, parents=True)

    # Data transformations
    data_transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize images
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    dataset = ChimpFaceDataset(root_dir=phase_data_path, transform=data_transform)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    # --- Model ---
    # Load the pre-trained model (Inception v3)
    model = models.inception_v3(weights=None)  # Don't load pre-trained weights here

    num_classes = max(ALL_CLASS_INDEX_TO_NAMES) + 1
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(Path(config.base_model) / MODEL_NAME))
    model.eval()  # Set the model to evaluation mode

    # create confusion matrix
    all_predictions = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for images, labels in tqdm(loader, total=len(loader)):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions = np.append(all_predictions, predicted.numpy())
            all_labels = np.append(all_labels, labels.numpy())

    # Initialize the confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for true_label, pred_label in zip(all_labels, all_predictions):
        conf_matrix[int(true_label), int(pred_label)] += 1

    def plot_confusion_matrix(matrix, class_names):
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Add labels to each cell
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                plt.text(j, i, str(matrix[i][j]), ha="center", va="center", color="red")

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

    class_names = [ALL_CLASS_INDEX_TO_NAMES[i] if i in ALL_CLASS_INDEX_TO_NAMES else f"Class {i}" for i in range(num_classes)]
    plot_confusion_matrix(conf_matrix, class_names)
    plt.savefig(output_path / f'{phase_name}_confusion_matrix.png')
    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=Path(r'D:\inference\faces_dataset_res'))
    parser.add_argument('--data_path', type=str, default=Path(r'D:\faces_dataset'))
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--base_model', type=str, default=r'D:\training_output\cloud\faces_classification\first_train')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_opt())
