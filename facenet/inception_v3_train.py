import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from pathlib import Path
import schedulefree
from torchvision.models import Inception_V3_Weights


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, person_folder in enumerate(self.root_dir.iterdir()):
            for image_path in person_folder.iterdir():
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations
from torchvision import transforms

data_transform = transforms.Compose([
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

# Create dataset
data_dir = Path(r'C:\Workspace\ChimpanzeesThesis\faces_images\individual_faces_dataset')
out_dir = Path(r'C:\Workspace\ChimpanzeesThesis\outputs\inception_v3_train')
full_dataset = FaceDataset(root_dir=data_dir, transform=data_transform)

# Split into train and validation sets
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)  # No need to shuffle validation data

# --- Model ---

# Load a pre-trained model (InceptionResnetV1)
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

# Modify the final classification layer
num_classes = len(list(data_dir.iterdir()))  # Number of people
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

load_best_model = True
if load_best_model:
    model.load_state_dict(torch.load(out_dir / 'best_face_recognition_model_with_augmentations.pth', weights_only=True))

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
        torch.save(model.state_dict(), out_dir / 'best_face_recognition_model_with_augmentations.pth')
        print("saving model...")

    # Training phase
    model.train()  # Set model to training mode
    optimizer.train()
    for images, labels in train_loader:
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