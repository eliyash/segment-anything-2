from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_embeddings(config, root_folder_path: Path, method: str = 'tsne', perplexity: int = 30):
    """
    Plots embeddings using t-SNE or PCA for dimensionality reduction.

    Args:
        config: training config
        root_folder_path (str): Path to all training files.
        method (str): Dimensionality reduction method ('tsne' or 'pca'). Default is 'tsne'.
        perplexity (int): Perplexity for t-SNE. Ignored if method is 'pca'. Default is 30.
    """
    # Load embeddings and labels
    embeddings = np.load(root_folder_path / config.output_labels)
    labels = np.load(root_folder_path / config.output_labels)

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        title = "t-SNE Visualization of Embeddings"
    elif method == 'pca':
        reducer = PCA(n_components=2)
        title = "PCA Visualization of Embeddings"
    else:
        raise ValueError("Invalid method. Use 'tsne' or 'pca'.")

    # Reduce dimensions
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7
    )
    plt.colorbar(scatter, label='Class Labels')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(root_folder_path / f'embeddings_{method}.png')
