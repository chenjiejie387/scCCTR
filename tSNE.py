from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from main import predicted_labels
class GeneExpressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
def generate_tsne_visualization(model, gene_expression_data, labels, scaler, device, file_path):
    # Standardize the entire dataset
    scaled_data = scaler.transform(gene_expression_data.values)

    # Construct a data loader for the entire dataset
    all_dataset = GeneExpressionDataset(scaled_data, labels)
    all_loader = DataLoader(all_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Extract features
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in all_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            features.extend(outputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    print(f"Number of features extracted for t-SNE: {len(features)}")

    # Perform dimensionality reduction using t-SNE and visualize
    tsne = TSNE(n_components=2, random_state=42)
    features_array = np.array(features)
    tsne_embeddings = tsne.fit_transform(features_array)

    # Custom color list, e.g., red, green, blue, yellow, etc.
    custom_palette = ['#ecd452', '#e76254', '#44757a', '#c6a468', '#6D4E7E',
                      '#2e5496', '#299d8f', '#4994c4', '#8f6d5f', '#FAC050',
                      '#77B473', '#CBC2D7', '#D28450', '#DCC2AB', '#90C3B6',
                      '#FDC192', '#3854A6', '#727174', '#F3756D', '#FAF49B',
                      '#9172DB']  # Adjust the number of colors based on the number of clusters

    # Check if there are enough colors to color each cluster
    num_clusters = np.unique(predicted_labels).max() + 1
    if len(custom_palette) < num_clusters:
        raise ValueError("The custom color list is not sufficient, please increase the number of colors.")

    # Map using the custom color list
    unique_clusters = sorted(np.unique(predicted_labels))
    color_map = dict(zip(unique_clusters, custom_palette))
    color_values = np.array([color_map[c] for c in predicted_labels])

    # Plot t-SNE results
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=color_values, s=20, alpha=1.0)
    plt.title('scCCTR')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Save the image
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()