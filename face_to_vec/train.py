import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple
from collections import defaultdict
import logging
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DataLoaderSlowSize
from torchvision import models, transforms

import random
from sklearn.model_selection import train_test_split
import argparse

from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from face_to_vec.plot_embedding import plot_embeddings


def setup_logger(output_folder):
    logger = logging.getLogger("ChimpFaceLogger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_folder / "training.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


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
    Classes with only one sample are excluded from being anchors and positives.
    They can still be used as negatives.
    """

    def __init__(self, file_paths: List[Path], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

        # Create a dictionary to map labels to indices
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Identify classes with at least two samples
        self.valid_labels = [label for label, indices in self.label_to_indices.items() if len(indices) > 1]
        if not self.valid_labels:
            raise ValueError("No classes with at least two samples available for triplet generation.")

        # Create a list of valid indices (only from classes with >=2 samples)
        self.valid_indices = [idx for idx, label in enumerate(labels) if label in self.valid_labels]

        # Precompute the list of all labels (including single-sample classes) for negatives
        self.all_labels = list(set(labels))

        # If needed, remove labels with only one sample from being selected as positives
        # This is already handled by valid_labels and valid_indices

    def __getitem__(self, index):
        anchor_idx = self.valid_indices[index]
        anchor_path = self.file_paths[anchor_idx]
        anchor_label = self.labels[anchor_idx]

        # Positive sample
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = random.choice(positive_indices)
        positive_path = self.file_paths[positive_idx]

        # Negative sample
        negative_label = random.choice(self.all_labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.all_labels)
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_path = self.file_paths[negative_idx]

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
        return len(self.valid_indices)


class DataLoader(DataLoaderSlowSize):
    def __init__(self, *args, **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)

        assert hasattr(self.dataset, '__len__'), "Dataset must implement __len__ method"
        self._len = getattr(self.dataset, '__len__')() // self.batch_size

    def __len__(self):
        return self._len


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

        transforms.RandomRotation(10),
        transforms.Resize((config.image_size, config.image_size)),
        # transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),

        transforms.Resize((config.image_size, config.image_size)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embed_dataset = ChimpFacesDataset(file_paths, labels, transform=embed_transform)
    embed_loader = DataLoader(embed_dataset, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)

    return embed_loader, labels


class EmbeddingNet(nn.Module):
    """
    Model that outputs normalized embeddings using a pre-trained ResNet backbone.
    """
    def __init__(self, embedding_dim: int, normalize: bool = True):
        super(EmbeddingNet, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, embedding_dim)
        # Optionally, add normalization
        self.normalize = normalize

    def forward(self, x):
        embedding = self.backbone(x)
        if self.normalize:
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


def train_epoch_triplet(
        model: nn.Module, device: torch.device, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn,
        logger, writer, epoch):
    """
    Trains the model for one epoch using triplet loss.
    """
    logger.info(f"Training epoch {epoch}...")
    model.train()
    running_loss = 0.0
    for batch_idx, (anchor, positive, negative) in tqdm(enumerate(dataloader), total=len(dataloader)):
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
    epoch_loss = running_loss / len(dataloader)
    logger.info(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    return epoch_loss


def validate_triplet(model: nn.Module, device: torch.device, dataloader: DataLoader, loss_fn, logger, writer, epoch):
    """
    Validates the model for one epoch using triplet loss.
    """
    logger.info(f"Validating epoch {epoch}...")
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, total=len(dataloader)):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            running_loss += loss.item() * anchor.size(0)
    epoch_loss = running_loss / len(dataloader)
    logger.info(f"Epoch {epoch} Validation Loss: {epoch_loss:.4f}")
    writer.add_scalar('Loss/Validation', epoch_loss, epoch)
    return epoch_loss


def extract_embeddings(model: nn.Module, device: torch.device, dataloader: DataLoader):
    """
    Extracts embeddings and corresponding labels from the dataset.
    """
    print("Extracting embeddings...")
    model.eval()
    embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu())
            all_labels.extend(labels)
    embeddings = torch.cat(embeddings)
    return embeddings.numpy(), np.array(all_labels)


def train_model_triplet(config, output_folder, device, logger):
    """
    Trains the model using triplet loss and saves the best model based on validation loss.
    """
    writer = SummaryWriter(log_dir=output_folder / 'tensorboard_logs')

    # Prepare data
    train_loader, val_loader, num_classes = prepare_triplet_data(config)
    logger.info(f"Number of classes: {num_classes}")

    # Initialize model
    model = EmbeddingNet(config.embedding_dim, config.normalize).to(device)

    # Define loss function
    loss_fn = nn.TripletMarginLoss(margin=config.margin, p=2)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch_triplet(model, device, train_loader, optimizer, loss_fn, logger, writer, epoch)
        val_loss = validate_triplet(model, device, val_loader, loss_fn, logger, writer, epoch)
        scheduler.step()

        logger.info(f"Epoch {epoch}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_folder / config.save_path)
            logger.info(f"Best model saved to {config.save_path}.")

    writer.close()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Chimp Face Identification with Triplet Loss')

    parser.add_argument('--output_folder', type=str, default='.', help='Path to the output directory')
    parser.add_argument('--data_dir', type=str, help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the output embeddings.')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size (image_size x image_size).')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for TripletMarginLoss.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    parser.add_argument('--device', default=None, help='Use GPU if available.')
    parser.add_argument('--save_path', type=str, default='best_triplet_model.pth', help='Path to save the best model.')
    parser.add_argument('--extract_embeddings', action='store_true', help='Flag to extract embeddings after training.')
    parser.add_argument('--output_embeddings', type=str, default='embeddings.npy', help='Path to save embeddings.')
    parser.add_argument('--output_labels', type=str, default='labels.npy', help='Path to save the labels.')
    parser.add_argument('--normalize', type=bool, default=True, help='Normalize embeddings to unit length.')

    args = parser.parse_args()
    return args


def main():
    config = parse_args()

    experiment_folder_name = f'{datetime.now():%Y_%m_%d__%H_%M_%S}'
    output_folder = Path(config.output_folder) / experiment_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_folder)
    logger.info("Starting training process...")
    logger.info(f"log: {output_folder.as_posix()}")

    (output_folder / 'config.json').write_text(json.dumps(config.__dict__, indent=4))
    device = torch.device(config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Train the model
    model = train_model_triplet(config, output_folder, device, logger)

    # If extraction is requested
    if config.extract_embeddings:
        # Load the best model
        model.load_state_dict(torch.load(output_folder / config.save_path))
        model.to(device)

        # Prepare embedding data
        embed_loader, labels = prepare_embedding_data(config)

        # Extract embeddings
        embeddings, labels = extract_embeddings(model, device, embed_loader)
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")

        # Save embeddings and labels
        np.save(output_folder / config.output_embeddings, embeddings)
        np.save(output_folder / config.output_labels, labels)
        logger.info(f"Embeddings saved to {config.output_embeddings}")
        logger.info(f"Labels saved to {config.output_labels}")
        plot_embeddings(config, output_folder, 'tsne')
        plot_embeddings(config, output_folder, 'pca')


if __name__ == '__main__':
    main()
