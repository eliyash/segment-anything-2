import argparse

import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import schedulefree
from torchvision.models import Inception_V3_Weights
from torchvision import transforms

from PIL import Image

from yolov9_main.monkey_names_with_classes import ALL_NAMES_TO_CLASS_INDEX


# Dataset definition
class ChimpFaceDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse folders to collect image paths and labels
        for index, individual_folder in enumerate(sorted(root_dir.iterdir())):
            if individual_folder.is_dir():
                label = ALL_NAMES_TO_CLASS_INDEX[individual_folder.name.split('_')[-1]]
                for img_file in individual_folder.iterdir():
                    self.image_paths.append(img_file)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def main(config, load_best_model=False):
    batch_size = config.batch_size
    data_path = Path(config.data_path)
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),

        # --- Standard Augmentations ---
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),  # Flip horizontally
            transforms.RandomRotation(degrees=15),  # Rotate by up to 15 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Translate and scale
        ], p=0.8),

        # --- Augmentations to reduce image quality ---
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
        ], p=0.8),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ChimpFaceDataset(root_dir=data_path / 'train', transform=train_transform)
    val_dataset = ChimpFaceDataset(root_dir=data_path / 'val', transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    # --- Model ---

    # Load a pre-trained model (InceptionResnetV1)
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

    # Modify the final classification layer
    num_classes = max(ALL_NAMES_TO_CLASS_INDEX) + 1  # Number of people
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if load_best_model and (output_dir / 'best_model.pth').exists():
        model.load_state_dict(torch.load(output_dir / 'best_model.pth', weights_only=True))

    # --- Training ---

    # Define optimizer and loss function
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = schedulefree.AdamWScheduleFree(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize variables to track best validation loss
    best_val_loss = float('inf')

    # Training loop
    num_epochs = 1000  # Adjust as needed
    for epoch in range(num_epochs):
        # Validation phase
        model.eval()   # Set model to evaluation mode
        optimizer.eval()
        val_loss = 0.0
        correct_classifications = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                # outputs = outputs.logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_classifications += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct_classifications / total_samples

        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%'
        )

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print("saving model...")

        # Training phase
        model.train()  # Set model to training mode
        optimizer.train()
        # i = 0
        for images, labels in train_loader:
            # i += 1
            # if i < 18 * 4 + 1:
            #     continue

            # Forward pass
            outputs = model(images)
            outputs = outputs.logits

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'.', end='')

    print("Training complete. Best model saved as 'best_face_recognition_model.pth'")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_opt())
