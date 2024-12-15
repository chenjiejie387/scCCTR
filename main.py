import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from AEmodel import VAE,AE,train_autoencoder
from sklearn.cluster import KMeans
import scanpy as sc
from prepross import _prepare_training_data
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
import argparse
from torch.utils.data import Dataset, DataLoader
from Transformer import (CellDataset,TransformerModel,validate_model,EarlyStopping,
                         evaluate_model,train_model,predict_model,predict_and_save,predict_full_dataset)
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from tSNE import GeneExpressionDataset
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def extract_top_30_percent_per_cluster(data, clusters, num_clusters):
    centroids = []
    reduced_data = data

    # Calculate the centroid of each cluster
    for i in range(num_clusters):
        cluster_points = reduced_data[clusters == i, :]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    new_matrices = []
    top_30_percent_indices = []

    # Extract the top 30% of points closest to the centroid for each cluster
    for i, centroid in enumerate(centroids):
        cluster_indices = np.where(clusters == i)[0]
        cluster_points = reduced_data[cluster_indices, :]

        # Calculate the Euclidean distance from each point to the centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        sorted_indices = np.argsort(distances)

        # Ensure at least one point is selected
        num_nearest_points = max(1, int(0.4 * len(cluster_indices)))
        nearest_points_indices = sorted_indices[:num_nearest_points]

        # Extract the top 30% of points
        top_30_percent_matrix = reduced_data[cluster_indices[nearest_points_indices], :]
        new_matrices.append(top_30_percent_matrix)
        top_30_percent_indices.append(cluster_indices[nearest_points_indices])

    return new_matrices, top_30_percent_indices

# Process each gene expression matrix by training an autoencoder and reconstructing the data
def process_gene_expression_matrices(gene_expression_matrices, epochs=100, learning_rate=0.0001):
    autoencoders = []
    reconstructed_matrices = []

    for i, matrix in enumerate(gene_expression_matrices):
        print(f'Training autoencoder for cluster {i}...')
        ae = AE(dim=matrix.shape[1])
        ae = train_autoencoder(ae, matrix, epochs=epochs, learning_rate=learning_rate)
        autoencoders.append(ae)

        # Reconstruct the gene expression matrix
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32)
        reconstructed_matrix, _ = ae(matrix_tensor)
        reconstructed_matrices.append(reconstructed_matrix.detach().numpy())

    return autoencoders, reconstructed_matrices

def replace_features_in_original_matrix(X_hvg, top_30_percent_indices, reconstructed_matrices):
    for indices, reconstructed_matrix in zip(top_30_percent_indices, reconstructed_matrices):
        X_hvg[indices, :] = reconstructed_matrix
    return X_hvg

def save_top_30_percent_cells_with_labels(cell_names, top_30_percent_indices, clusters, output_path):

    selected_cells = []
    selected_labels = []

    for indices in top_30_percent_indices:
        for idx in indices:
            selected_cells.append(cell_names[idx])
            selected_labels.append(clusters[idx])

    result_df = pd.DataFrame({'Cell_Name': selected_cells, 'Cluster': selected_labels})

    result_df.to_csv(output_path, index=False)

    print(f"CSV file saved to {output_path}")

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy with true labels.
    Args:
        y_true: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
    Returns:
        accuracy: accuracy, in [0,1]
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

class CellDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def Iterative_selecting(
    data_file_path,
    clusters_file_path,
    hvg_save_path,
    final_hvg_save_path,
    latent_embeddings_save_path,
    final_clusters_save_path,
    top_30_percent_save_path,
    num_clusters=8,
    max_iterations=6,
    vae_params=None,
    vae_training_epochs=1200,
    ae_epochs=120,
    ae_learning_rate=0.0001,
    device=torch.device("cuda:1")
):
    if vae_params is None:
        vae_params = {
            "num_genes": 2000,
            "latent_dim": 64,
            "hidden_dim": [256, 128, 64],
            "decoder_activation": "nn.linear",
        }

    # Load data
    data = pd.read_csv(data_file_path, index_col=0)
    data_filled = data.fillna(value=0)
    adata = sc.AnnData(data_filled.T)

    # Prepare training data
    adata_hvg, X_hvg = _prepare_training_data(adata)
    print(f'X_hvg shape: {X_hvg.shape}')

    cell_names = adata_hvg.obs_names
    gene_names = adata_hvg.var_names

    # Save initial gene selection results
    X_hvg_df = pd.DataFrame(X_hvg, index=cell_names, columns=gene_names)
    X_hvg_df.to_csv(hvg_save_path)
    print(f'Saved X_hvg to {hvg_save_path}')

    # Read true cluster information
    df1 = pd.read_csv(clusters_file_path)
    true_cluster = df1['Cluster']

    all_clusterings = []
    previous_clusters = None

    # Initialize VAE
    vae = VAE(
        num_genes=vae_params["num_genes"],
        latent_dim=vae_params["latent_dim"],
        device=device,
        hidden_dim=vae_params["hidden_dim"],
        decoder_activation=vae_params["decoder_activation"],
    ).to(device)
    all_ari_values = []
    all_acc_values = []
    all_nmi_values = []
    # Iterative selection process
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")

        # VAE training
        data_tensor = torch.tensor(X_hvg, dtype=torch.float32).to(device)
        for epoch in range(vae_training_epochs):
            losses = vae.train_vae(data_tensor)
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{vae_training_epochs}], reconstruction_loss: {losses["loss_reconstruction"]:.4f}')

        # Extract latent space
        vae.eval()
        with torch.no_grad():
            data_z = vae.encoder(data_tensor)
        data_z_cpu = data_z.cpu().numpy()

        # KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
        clusters = kmeans.fit_predict(data_z_cpu)

        all_clusterings.append(clusters)

        # Build consensus matrix
        consensus_matrix = np.zeros((data_z_cpu.shape[0], data_z_cpu.shape[0]))
        for clustering in all_clusterings:
            for i in range(data_z_cpu.shape[0]):
                for j in range(data_z_cpu.shape[0]):
                    if clustering[i] == clustering[j]:
                        consensus_matrix[i, j] += 1

        consensus_matrix /= len(all_clusterings)
        distance_matrix = 1 - consensus_matrix

        # Hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
        stable_clusters = clustering.fit_predict(distance_matrix)
        all_clusterings[-1] = stable_clusters


        if previous_clusters is not None:
            ari_change = adjusted_rand_score(previous_clusters, stable_clusters)
            print(f"ARI between current and previous stable_clusters: {ari_change}")
            if ari_change > 0.95:
                print("ARI difference is less than 0.05, stopping iterations early.")
                break

        previous_clusters = stable_clusters

        # Extract the top 30% of node features closest to the centroid
        new_matrices, top_30_percent_indices = extract_top_30_percent_per_cluster(X_hvg, stable_clusters, num_clusters)
        autoencoders, reconstructed_matrices = process_gene_expression_matrices(new_matrices, ae_epochs, ae_learning_rate)
        X_hvg = replace_features_in_original_matrix(X_hvg, top_30_percent_indices, reconstructed_matrices)

    # Save results
    X_hvg_df = pd.DataFrame(X_hvg, index=cell_names, columns=gene_names)
    X_hvg_df.to_csv(final_hvg_save_path)
    print(f'Saved final X_hvg to {final_hvg_save_path}')

    latent_embeddings_df = pd.DataFrame(data_z_cpu)
    latent_embeddings_df.to_csv(latent_embeddings_save_path, index=False)
    print(f"Latent embeddings saved to {latent_embeddings_save_path}")

    results_df = pd.DataFrame({'cell_name': cell_names, 'Cluster': stable_clusters})
    results_df.to_csv(final_clusters_save_path, index=False)
    print(f"Final clusters saved to {final_clusters_save_path}")

    save_top_30_percent_cells_with_labels(cell_names, top_30_percent_indices, stable_clusters, top_30_percent_save_path)
    print(f"Top 30% cells with labels saved to {top_30_percent_save_path}")



def predict_model(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())

    return predictions

def predict_and_save(model, data_loader, gene_expression_data, file_path, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    test_cell_names = gene_expression_data.index[data_loader.dataset.indices]

    predicted_clusters_df = pd.DataFrame({
        'Cell Name': test_cell_names,
        'Predicted Cluster': predictions
    })

    predicted_clusters_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

def predict_full_dataset(model, data, cell_names, file_path, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in DataLoader(data, batch_size=32):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    predicted_clusters_df = pd.DataFrame({
        'Cell Name': cell_names,
        'Predicted Cluster': predictions
    })
    predicted_clusters_df.to_csv(file_path, index=False)

def train_and_predict_model(args):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    gene_expression_data = pd.read_csv(args.final_hvg_save_path, index_col=0).transpose()

    cells_data = pd.read_csv(args.top_30_percent_save_path)
    train_cell_names = cells_data['Cell_Name'].values
    y_train_labels = cells_data['Cluster'].values

    unique_clusters_cells_data = cells_data['Cluster'].unique()
    print("Unique clusters in cells_data:", unique_clusters_cells_data)

    X_train = gene_expression_data.loc[train_cell_names].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    input_dim = gene_expression_data.shape[1]
    model = TransformerModel(input_dim=input_dim, num_classes=len(set(y_train_labels)), num_heads=args.num_heads, num_layers=args.num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = GeneExpressionDataset(X_train, y_train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    num_epochs = args.num_epochs
    num_epochs_earlystop = args.num_epochs_earlystop

    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=True)

    for epoch in range(num_epochs_earlystop):
        train_model(model, train_loader, optimizer, criterion, num_epochs, device)
        val_loss = validate_model(model, train_loader, criterion, device)
        if early_stopping(val_loss, model):
            print("Early stopping triggered")
            break

    scaled_data = scaler.transform(gene_expression_data.values)
    predict_full_dataset(model, torch.tensor(scaled_data, dtype=torch.float32).to(device), gene_expression_data.index,
                         args.predicted_clusters_save_path, device)


    model.eval()
    with torch.no_grad():
        scaled_data = scaler.transform(gene_expression_data.values)
        predictions = model(torch.tensor(scaled_data, dtype=torch.float32).to(device))
        predicted_labels = predictions.argmax(dim=1).cpu().numpy()
    return predicted_labels

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Iterative Selection-based Clustering for Single-cell RNA-seq',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the input data CSV file.')
    parser.add_argument('--clusters_file_path', type=str, required=True, help='Path to the true clusters CSV file.')
    parser.add_argument('--hvg_save_path', type=str, required=True, help='Path to save the high variance genes file.')
    parser.add_argument('--final_hvg_save_path', type=str, required=True, help='Path to save the final X_hvg file.')
    parser.add_argument('--latent_embeddings_save_path', type=str, required=True,
                        help='Path to save latent embeddings.')
    parser.add_argument('--final_clusters_save_path', type=str, required=True, help='Path to save final clusters.')
    parser.add_argument('--top_30_percent_save_path', type=str, required=True,
                        help='Path to save top 30% cells labels.')
    parser.add_argument('--num_clusters', type=int, default=8, help='Number of clusters for clustering.')
    parser.add_argument('--max_iterations', type=int, default=6, help='Maximum number of iterations.')
    parser.add_argument('--vae_training_epochs', type=int, default=1200, help='Number of epochs for VAE training.')
    parser.add_argument('--ae_epochs', type=int, default=120, help='Number of epochs for autoencoder training.')
    parser.add_argument('--ae_learning_rate', type=float, default=0.0001,
                        help='Learning rate for autoencoder training.')

    parser.add_argument('--cells_file_path', type=str, required=True, help='Path to the cells data CSV file.')
    parser.add_argument('--predicted_clusters_save_path', type=str, required=True,
                        help='Path to save predicted clusters.')

    # 模型参数
    parser.add_argument('--num_heads', type=int, default=32, help='Number of heads in the Transformer model.')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in the Transformer model.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training.')
    parser.add_argument('--num_epochs_earlystop', type=int, default=50, help='Number of epochs for early stopping.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--delta', type=float, default=1e-2, help='Delta for early stopping.')
    args = parser.parse_args()

    Iterative_selecting(
        data_file_path=args.data_file_path,
        clusters_file_path=args.clusters_file_path,
        hvg_save_path=args.hvg_save_path,
        final_hvg_save_path=args.final_hvg_save_path,
        latent_embeddings_save_path=args.latent_embeddings_save_path,
        final_clusters_save_path=args.final_clusters_save_path,
        top_30_percent_save_path=args.top_30_percent_save_path,
        num_clusters=args.num_clusters,
        max_iterations=args.max_iterations,
        vae_params={
            "num_genes": 2000,
            "latent_dim": 64,
            "hidden_dim": [256, 128, 64],
            "decoder_activation": "nn.linear",
        },
        vae_training_epochs=args.vae_training_epochs,
        ae_epochs=args.ae_epochs,
        ae_learning_rate=args.ae_learning_rate,
        device=torch.device("cuda:1"),
    )
    predicted_labels = train_and_predict_model(args)


