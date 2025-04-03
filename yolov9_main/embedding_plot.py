import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap


def draw_embeddings(embedding_list, output_path):
    # Aggregate all embeddings and keep track of which frame each embedding belongs to.
    all_embeddings = []
    frame_indices = []  # This list will store the frame index for each embedding.
    for frame_index, frame_embeddings in enumerate(embedding_list):
        for emb in frame_embeddings:
            all_embeddings.append(emb)
            frame_indices.append(frame_index)

    all_embeddings = np.array(all_embeddings)
    frame_indices = np.array(frame_indices)

    # Run t-SNE to reduce embeddings to 2 dimensions.
    tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Create a custom colormap that goes from green -> yellow -> orange -> red.
    # We use the frame indices to pick colors.
    num_frames = len(embedding_list)
    custom_colors = ['green', 'yellow', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors, N=num_frames)

    # Map each frame index to a color in the gradient.
    point_colors = [cmap(i / (num_frames - 1)) for i in frame_indices]

    # Plot the t-SNE results.
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=point_colors, alpha=0.7)
    plt.title('t-SNE Embeddings of Face Features\n(Color Gradient by Frame Index)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(output_path)