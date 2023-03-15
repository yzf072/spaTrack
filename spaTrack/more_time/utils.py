import numpy as np
from scipy.stats import wasserstein_distance
from scipy.sparse import issparse
import ot
from anndata import AnnData

#### Calculate spatial coordinates distances
def spatial_dist(adata:AnnData, spatial_key:str='spatial',spa_method:str='euclidean'):
    """Calculate spatial distance.
    Args:
        adata:AnnData object.
        spatial_key: adata.obsm[spatial_key],corresponds to spatial coordinates.
        metric
    Rerurns:
        spatial distances. The dimension is n_obs*n_obs.
    """
    spa_coords = adata.obsm[spatial_key]
    spa_dist = ot.dist(spa_coords, spa_coords, metric=spa_method)
    return np.array(spa_dist) 


#### Get gene expression matrix from adata
def get_exp_matrix(adata:AnnData,layer:str ='X'):
    """Get gene expression matrix(dense not sparse) from adata object.
    Args:
        adata:AnnData object.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``adata.layers[layer]``.
    Rerurns:
        gene expression matrix. The dimension is n_obs*n_genes.
    """
    if layer == 'X':
        exp_matrix = adata.X
    else:
        exp_matrix = adata.layers[layer]
    if issparse(exp_matrix):
        exp_matrix = exp_matrix.toarray()
    else:
        exp_matrix = np.array(exp_matrix)
    return exp_matrix


#### Calculate gene expression dissimilarity

def wasserstein_distance(
    X:np.ndarray,
    Y:np.ndarray,
):
    """Compute Wasserstein distance between two gene expression matrix.
    
    Args:
        X: np.array with dim (n_obs * n_genes).
        Y: np.array with dim (m_obs * n_genes).
    Rerurns:
        W_D: np.array with dim(n_obs * m_obs). Wasserstein distance matrix.
        
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    W_D = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dist = wasserstein_distance(X[i], Y[j])
            dist_matrix[i][j] = dist
    return W_D

def kl_divergence_backend(
    X:np.ndarray,
    Y:np.ndarray,
):
    """Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    Takes advantage of POT backend to speed up computation.
    
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)

    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum("ij,ij->i", X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    KL_D = X_log_X.T - nx.dot(X, log_Y.T)
    return KL_D

def mcc_distance(
    X:np.ndarray,
    Y:np.ndarray,
):
    
    """Compute matthew's correlation coefficient between two gene expression matrix.
    Args:
        X: np.array with dim (n_obs * n_genes).
        Y: np.array with dim (m_obs * n_genes).
    
    """
    def __mcc(true_labels, pred_labels):
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
        mcc = (TP * TN) - (FP * FN)
        denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        if denom==0:
            return 0
        return mcc / denom
    
    cost = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            cost[i, j] = __mcc(X[i], Y[j])
    return cost



def gene_dist(
    X_1:np.ndarray,
    X_2:np.ndarray,
    gene_method:str='kl',
):
    """Calculate gene expression dissimilarity.
    
    Args:
        X: Gene expression matrix of adata_1. np.array with dim (n_obs * n_genes).
        Y: Gene expression matrix of adata_2. np.array with dim (n_obs * n_genes).
        method: calculate gene expression dissimilarity measure: ``'euclidean'`` or ``'cosine'``or``'mcc'``or``'wasserstein'``or``'kl'``.
    Rerurns:
        W_D: np.array with dim(n_obs * m_obs). Wasserstein distance matrix.
        
    """
    if gene_method == 'euclidean' or gene_method == 'cosine':
        dist_matrix = ot.dist(X_1,X_2,metric=gene_method)
    elif gene_method == 'mcc':
        dist_matrix = mcc_distance(X_1,X_2)

    elif gene_method == 'wasserstein':
        dist_matrix = wasserstein_distance(X_1,X_2) 
                
    elif gene_method == 'kl':
        s_X_1 = X_1 + 0.01
        s_X_2 = X_2 + 0.01
        dist_matrix = (kl_divergence_backend(s_X_1, s_X_2) + kl_divergence_backend(s_X_2, s_X_1).T) / 2

    return dist_matrix

##check adata type
def pre_check_adata(
    adata:AnnData,
    spatial_key:str='spatial',
    time:str='time',
    annotation:str='celltype'
):
    """
    Check adata type. 
    """
    adata.obs['time'] = adata.obs[time]
    adata.obs['annotation'] = adata.obs[annotation]
    adata.obs['x'] = adata.obsm[spatial_key][:,0]
    adata.obs['y'] = adata.obsm[spatial_key][:,1]
    adata.obs['cell_id'] = adata.obs.index
    return adata
