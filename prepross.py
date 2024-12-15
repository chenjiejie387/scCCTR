import numpy as np
import scanpy as sc

def _prepare_training_data(adata):
    print('Preparing training data...')

    # Copy the data for preprocessing
    adata_pp = adata.copy()

    # Start preprocessing steps
    print('Applying raw counts preprocessing...')

    # Convert data type to float
    adata_pp.X = adata_pp.X.astype(float)

    # Normalize total counts
    sc.pp.normalize_total(adata_pp)

    # Logarithmize the data
    sc.pp.log1p(adata_pp)

    # Check for and replace NaN values
    adata_pp.X = np.nan_to_num(adata_pp.X, nan=0.0)

    # Calculate the variance of each gene and select the genes with the highest variance
    num_selected_genes = 2000
    gene_variances = np.var(adata_pp.X, axis=0)
    selected_genes = gene_variances.argsort()[-num_selected_genes:]

    # Update adata_pp with the selected gene indices
    adata_hvg = adata_pp[:, selected_genes]
    X_hvg = adata_hvg.X.astype(np.float32)  # Enforce conversion to Float type

    print(f'HVG adata shape: {adata_hvg.shape}')

    return adata_hvg, X_hvg