import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
from sklearn.metrics import confusion_matrix
import wandb

def compute_confusion_matrix(resnet, val_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_labels, all_preds

def main():
    # Initialize W&B
    # wandb.init(project="your_project_name", name="detection_logging")

    is_windows = os.name == 'nt'
    root_dir = Path(r'C:\Workspace\ChimpanzeesThesis\faces_images') if is_windows else Path(r'/home/ubuntu/faces_work')
    data_dir = root_dir / 'individual_faces_dataset'
    out_dir = root_dir / 'training'

    batch_size = 16
    epochs = 10
    workers = 0 if is_windows else 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    train_name = datetime.now().strftime('train__%Y%m%d_%H%M%S')
    model_folder = out_dir / train_name
    model_folder.mkdir(parents=True, exist_ok=True)

    log_file_path = model_folder / 'training_logs.json'
    logs = []

    # Augmentation pipeline for training
    train_transforms = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        fixed_image_standardization
    ])

    val_transforms = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        fixed_image_standardization
    ])

    train_dataset = datasets.ImageFolder(data_dir.as_posix(), transform=train_transforms)
    val_dataset = datasets.ImageFolder(data_dir.as_posix(), transform=val_transforms)

    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(train_dataset.class_to_idx)
    ).to(device)

    resnet.load_state_dict(torch.load(r"C:\Workspace\ChimpanzeesThesis\outputs\facenet_train__20240920_211541\model_best.pt"), strict=False)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    img_inds = np.arange(len(train_dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        train_dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    best_val_loss = None
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Train phase
        resnet.train()
        train_loss, train_metrics = training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device
        )

        # Validation phase
        resnet.eval()
        validation_loss, validation_metrics = training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device
        )

        # Compute confusion matrix for validation
        cm, all_labels, all_preds = compute_confusion_matrix(resnet, val_loader, device)

        # Log to W&B: Confusion Matrix, Train & Validation Loss/Accuracy
        class_names = list(train_dataset.class_to_idx.keys())
        # wandb.log({
        #     "train_loss": train_loss,
        #     "val_loss": validation_loss,
        #     "train_accuracy": train_metrics['acc'],
        #     "val_accuracy": validation_metrics['acc'],
        #     "confusion_matrix": wandb.plot.confusion_matrix(
        #         probs=None,
        #         y_true=all_labels,
        #         preds=all_preds,
        #         class_names=class_names
        #     )
        # })

        # Save logs to file
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': validation_loss,
            'train_accuracy': train_metrics['acc'],
            'val_accuracy': validation_metrics['acc'],
        }
        logs.append({k: float(v) for k, v in epoch_log.items()})
        with open(log_file_path, 'w') as log_file:
            json.dump(logs, log_file, indent=4)

        # Save model checkpoints
        if not best_val_loss or validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(resnet.state_dict(), model_folder / 'model_best.pt')

        torch.save(resnet.state_dict(), model_folder / 'model_last.pt')

    # End W&B run
    # wandb.finish()

if __name__ == '__main__':
    main()
