from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import random
from sklearn.model_selection import train_test_split
import argparse


class ChimpFacesDataset(Dataset):
    """
    Dataset for loading images and labels, used for embedding extraction.
    """
    def __init__(self, file_paths: List[Path], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        # Read image using cv2
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Image not found or unable to read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label


class TripletDataset(Dataset):
    """
    Dataset for generating triplets for Triplet Margin Loss.
    """
    def __init__(self, file_paths: List[Path], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        # Create a dictionary to map labels to indices
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.labels_set = list(set(labels))

    def __getitem__(self, index):
        anchor_path = self.file_paths[index]
        anchor_label = self.labels[index]
        # Positive sample
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.label_to_indices[anchor_label])
        positive_path = self.file_paths[positive_index]
        # Negative sample
        negative_label = random.choice(self.labels_set)
        while negative_label == anchor_label:
            negative_label = random.choice(self.labels_set)
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_path = self.file_paths[negative_index]

        # Load images
        anchor_image = cv2.imread(str(anchor_path))
        if anchor_image is None:
            raise ValueError(f"Image not found or unable to read: {anchor_path}")
        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)

        positive_image = cv2.imread(str(positive_path))
        if positive_image is None:
            raise ValueError(f"Image not found or unable to read: {positive_path}")
        positive_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2RGB)

        negative_image = cv2.imread(str(negative_path))
        if negative_image is None:
            raise ValueError(f"Image not found or unable to read: {negative_path}")
        negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

    def __len__(self):
        return len(self.file_paths)


def prepare_triplet_data(config) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepares DataLoaders for training and validation using TripletDataset.
    """
    data_dir = Path(config.data_dir)
    # Gather all image paths and labels
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    file_paths = []
    labels = []
    for cls in classes:
        cls_dir = data_dir / cls
        for img_path in cls_dir.glob('*'):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                file_paths.append(img_path)
                labels.append(class_to_idx[cls])

    # Check if any class has less than 2 samples
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[label] += 1
    for label, count in class_counts.items():
        if count < 2:
            raise ValueError(f"Class {label} has less than 2 samples, cannot form triplets.")

    # Split into training and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=config.val_split, stratify=labels, random_state=config.random_seed
    )

    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create Triplet Datasets
    train_dataset = TripletDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = TripletDataset(val_paths, val_labels, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    num_classes = len(classes)
    return train_loader, val_loader, num_classes


def prepare_embedding_data(config) -> Tuple[DataLoader, List[int]]:
    """
    Prepares DataLoader for embedding extraction.
    """
    data_dir = Path(config.data_dir)
    # Gather all image paths and labels
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    file_paths = []
    labels = []
    for cls in classes:
        cls_dir = data_dir / cls
        for img_path in cls_dir.glob('*'):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                file_paths.append(img_path)
                labels.append(class_to_idx[cls])

    # Define transformations (no augmentation, just normalization)
    embed_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    embed_dataset = ChimpFacesDataset(file_paths, labels, transform=embed_transform)
    embed_loader = DataLoader(embed_dataset, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)

    return embed_loader, labels


class EmbeddingNet(nn.Module):
    """
    Model that outputs normalized embeddings using a pre-trained ResNet backbone.
    """
    def __init__(self, embedding_dim: int):
        super(EmbeddingNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        # Replace the final layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
        # Optionally, add normalization
        self.normalize = True

    def forward(self, x):
        embedding = self.backbone(x)
        if self.normalize:
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


def train_epoch_triplet(
        model: nn.Module, device: torch.device, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn
):
    """
    Trains the model for one epoch using triplet loss.
    """
    model.train()
    running_loss = 0.0
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        optimizer.zero_grad()
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * anchor.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_triplet(model: nn.Module, device: torch.device, dataloader: DataLoader, loss_fn):
    """
    Validates the model for one epoch using triplet loss.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            running_loss += loss.item() * anchor.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def extract_embeddings(model: nn.Module, device: torch.device, dataloader: DataLoader):
    """
    Extracts embeddings and corresponding labels from the dataset.
    """
    model.eval()
    embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu())
            all_labels.extend(labels)
    embeddings = torch.cat(embeddings)
    return embeddings.numpy(), labels


def train_model_triplet(config, device):
    """
    Trains the model using triplet loss and saves the best model based on validation loss.
    """
    # Prepare data
    train_loader, val_loader, num_classes = prepare_triplet_data(config)
    print(f"Number of classes: {num_classes}")

    # Initialize model
    model = EmbeddingNet(config.embedding_dim).to(device)

    # Define loss function
    loss_fn = nn.TripletMarginLoss(margin=config.margin, p=2)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch_triplet(model, device, train_loader, optimizer, loss_fn)
        val_loss = validate_triplet(model, device, val_loader, loss_fn)
        scheduler.step()

        print(f"Epoch {epoch}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.save_path)
            print(f"Best model saved to {config.save_path}.")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Chimp Face Identification with Triplet Loss')

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the output embeddings.')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size (image_size x image_size).')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for TripletMarginLoss.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--save_path', type=str, default='best_triplet_model.pth', help='Path to save the best model.')
    parser.add_argument('--extract_embeddings', action='store_true', help='Flag to extract embeddings after training.')
    parser.add_argument('--output_embeddings', type=str, default='embeddings.npy', help='Path to save embeddings.')
    parser.add_argument('--output_labels', type=str, default='labels.npy', help='Path to save the labels.')

    args = parser.parse_args()
    return args


def main():
    config = parse_args()

    # Set device
    device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Train the model
    model = train_model_triplet(config, device)

    # If extraction is requested
    if config.extract_embeddings:
        # Load the best model
        model.load_state_dict(torch.load(config.save_path))
        model.to(device)

        # Prepare embedding data
        embed_loader, labels = prepare_embedding_data(config)

        # Extract embeddings
        embeddings, labels = extract_embeddings(model, device, embed_loader)
        print(f"Extracted embeddings shape: {embeddings.shape}")

        # Save embeddings and labels
        np.save(config.output_embeddings, embeddings)
        np.save(config.output_labels, labels)
        print(f"Embeddings saved to {config.output_embeddings}")
        print(f"Labels saved to {config.output_labels}")


if __name__ == '__main__':
    main()
