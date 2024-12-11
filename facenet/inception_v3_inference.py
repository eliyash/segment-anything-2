import torch
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
from PIL import Image


def predict_class(image_np, data_transform, model):
    """Predicts the class index of a single image (NumPy array)."""

    image = Image.fromarray(image_np).convert('RGB')  # Convert NumPy array to PIL Image
    image = data_transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

def predict_classes(image_nps):
    """
    Predicts classes for a list of images (NumPy arrays).

    Args:
      image_nps: A list of NumPy array images.

    Returns:
      A list of predicted class names.
    """
    # Data transformations
    data_transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize images
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # --- Model ---
    # Load the pre-trained model (Inception v3)
    model = models.inception_v3(weights=None)  # Don't load pre-trained weights here

    # --- Get the training data directory ---
    # (You'll need this to get the class names)
    out_dir = Path(r'C:\Workspace\ChimpanzeesThesis\outputs\inception_v3_train')
    data_dir = Path(r'C:\Workspace\ChimpanzeesThesis\faces_images\individual_faces_dataset')

    # --- Get the number of classes and create the final classification layer ---
    num_classes = len(list(data_dir.iterdir()))  # Number of people
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(out_dir / 'best_face_recognition_model.pth'))
    model.eval()  # Set the model to evaluation mode

    # --- Create the class name mapping ---
    class_names = [p.name for p in data_dir.iterdir()]  # Get class names from folders

    predicted_class_names = []

    for image_np in image_nps:
        class_index = predict_class(image_np, data_transform, model)
        predicted_class_names.append(class_names[class_index])  # Map index to name

    return predicted_class_names
#
# # Example usage:
# # Assuming you have a list of NumPy array images called 'image_list'
# image_list = [
# ]
#
# predicted_class_names = predict_classes(image_list)
# print(predicted_class_names)