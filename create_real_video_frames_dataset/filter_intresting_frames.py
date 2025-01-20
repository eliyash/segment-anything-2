import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def extract_features(model, preprocess, image_path_batch):
    print(f'Extracting features from {image_path_batch}')

    def load_and_preprocess_image(path):
        return preprocess(Image.open(path).convert("RGB"))

    # Load images in parallel
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_and_preprocess_image, image_path_batch))

    input_tensor = torch.stack(images)

    output_batch = model(input_tensor.cuda()).cpu()

    features = [per_image_features.flatten().numpy() for per_image_features in output_batch]
    return features


# Compute pairwise distances
def compute_distances(feature_list):
    feature_matrix = np.vstack(feature_list)  # Stack feature vectors into a matrix
    return cosine_distances(feature_matrix)  # Calculate pairwise cosine distances


# Select the most different frames
def select_different_frames(distances, num_frames):
    selected_indices = [0]  # Start with the first frame
    for _ in range(1, num_frames):
        # Find the frame with the maximum minimum distance to the selected frames
        min_distances = distances[selected_indices].min(axis=0)
        next_index = min_distances.argmax()
        selected_indices.append(int(next_index))
    return selected_indices


def main():
    # List of frame file paths
    dataset_folder = Path(r"D:/frames_collection_per_signal_fixed_interlacing")
    output_folder = Path(r"D:/frames_collection_per_signal_fixed_interlacing_filtered")
    output_folder.mkdir(exist_ok=True)
    frames = sorted([path for path in dataset_folder.iterdir() if path.suffix in ['.jpg', '.png']])
    number_of_frames = len(frames)
    batch_size = 128
    num_frames_to_select = 2000

    # Load pre-trained VGG16 model
    vgg = models.vgg16(pretrained=True)
    vgg = torch.nn.Sequential(*list(vgg.children())[:-1]).cuda()
    vgg.eval()

    # Define image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract features from all frames
    print("Extracting features...")
    features = []
    with torch.no_grad():
        frames_batches = [frames[i:min(i + batch_size, number_of_frames)] for i in range(0, number_of_frames, batch_size)]
        for frames_batch in frames_batches:
            features.extend(extract_features(vgg, preprocess, frames_batch))

    np.save(output_folder / 'features.npy', np.array(features))

    features = np.load(output_folder / 'features.npy')

    # Compute distances between all pairs of frames
    print("Computing pairwise distances...")
    distances = compute_distances(features)

    print(f"Selecting {num_frames_to_select} most different frames...")
    most_different_indices = select_different_frames(distances, num_frames_to_select)

    # Get the paths of the selected frames
    selected_frames = [frames[i].name for i in sorted(most_different_indices)]
    (dataset_folder / 'chosen.json').write_text(json.dumps(selected_frames, indent=4))
    print("Most different frames:", selected_frames)

    for image_name in json.loads((dataset_folder / 'chosen.json').read_text()):
        # copy image from dataset_folder to output_folder
        shutil.copy(dataset_folder / image_name, output_folder / image_name)


# Main function
if __name__ == "__main__":
    main()
