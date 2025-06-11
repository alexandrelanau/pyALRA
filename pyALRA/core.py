import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD

def normalize_data(A):
    """
    Normalize data by library size and log-transform.
    A: Input matrix with cells as rows and genes as columns.
    """
    total_umi_per_cell = A.sum(axis=1)
    if np.any(total_umi_per_cell == 0):
        non_zero_idx = np.where(total_umi_per_cell > 0)[0]
        A = A[non_zero_idx, :]
        total_umi_per_cell = total_umi_per_cell[non_zero_idx]
        print(f"Removed {len(non_zero_idx)} cells which did not express any genes")

    A_norm = A / total_umi_per_cell[:, np.newaxis]
    A_norm *= 1e4
    A_norm = np.log1p(A_norm)

    return A_norm


def randomized_svd_py(A, K, q, random_state, type=None):
    """
    Perform SVD with an option for randomized or truncated SVD.

    Parameters:
    A : ndarray
        Input data matrix to decompose.
    K : int
        Number of singular values and vectors to compute.
    q : int
        Number of power iterations (only applicable for randomized SVD).
    random_state : int
        Random seed for reproducibility.
    type : str, optional
        If 'truncated', use TruncatedSVD; otherwise, use randomized SVD.
    
    Returns:
    U : ndarray
        Left singular vectors.
    Sigma : ndarray
        Singular values.
    VT : ndarray
        Right singular vectors transposed.
    """
    if type == 'truncated':
        svd = TruncatedSVD(n_components=K, random_state=random_state,n_iter=q)
        U = svd.fit_transform(A)
        Sigma = svd.singular_values_
        VT = svd.components_
    else:
        U, Sigma, VT = randomized_svd(A, n_components=K, n_iter=q, random_state=random_state)
    
    return U, Sigma, VT

def choose_k(A_norm, K=100, thresh=6, noise_start=80, q=12,random_state=1,type=None):
    if K > min(A_norm.shape):
        raise ValueError("K must be smaller than the smallest dimension of A_norm.")
    
    if noise_start > K - 5:
        raise ValueError("There need to be at least 5 singular values considered noise.")
    
    noise_svals = np.arange(noise_start, K)
    
    U, D, VT = randomized_svd_py(A_norm, K, q=q,random_state=random_state,type=type)

    # Calculate the differences between consecutive singular values
    diffs = D[:-1] - D[1:]
    
    # Calculate mean and standard deviation of noise singular value differences
    mu = np.mean(diffs[noise_svals - 1])
    sigma = np.std(diffs[noise_svals - 1])
    
    # Calculate the number of standard deviations from the mean
    num_of_sds = (diffs - mu) / sigma
    
    # Find the largest k where num_of_sds exceeds the threshold
    k = np.max(np.where(num_of_sds > thresh)[0]) + 1  # Adjust index for Python's 0-based indexing
    
    return {'k': k, 'num_of_sds': num_of_sds, 'd': D}


def alra(A_norm, k=0, q=12, quantile_prob=0.001,random_state=1,type=None):
    """
    ALRA: Adaptive thresholded low-rank approximation.
    """
    print(f"Read matrix with {A_norm.shape[0]} cells and {A_norm.shape[1]} genes")


    if k == 0:
        k_choice = choose_k(A_norm, q=q,random_state=random_state,type=type)
        k = k_choice['k']
        print(f"Chose k={k}")

    if sp.issparse(A_norm):
        A_norm=A_norm.toarray()
        
    originally_nonzero = A_norm > 0
    U, Sigma, VT = randomized_svd_py(A_norm, k, q=q,random_state=random_state,type=type)

    A_norm_rank_k = np.dot(U[:, :k], np.dot(np.diag(Sigma[:k]), VT[:k, :]))

    print(f"Find the {quantile_prob} quantile of each gene")
    A_norm_rank_k_mins = np.abs(np.quantile(A_norm_rank_k, quantile_prob, axis=0))
    
    print("Sweep")
    A_norm_rank_k_cor = np.where(A_norm_rank_k <= A_norm_rank_k_mins[np.newaxis, :], 0, A_norm_rank_k)

    def sd_nonzero(x):
        return np.std(x[x != 0])
    
    sigma_1 = np.apply_along_axis(sd_nonzero, 0, A_norm_rank_k_cor)
    sigma_2 = np.apply_along_axis(sd_nonzero, 0, A_norm)
    mu_1 = np.sum(A_norm_rank_k_cor, axis=0) / np.sum(A_norm_rank_k_cor != 0, axis=0)
    mu_2 = np.sum(A_norm, axis=0) / np.sum(A_norm != 0, axis=0)

    toscale = (~np.isnan(sigma_1)) & (~np.isnan(sigma_2)) & ~((sigma_1 == 0) & (sigma_2 == 0)) & (sigma_1 != 0)

    print(f"Scaling all except for {np.sum(~toscale)} columns")

    sigma_1_2 = sigma_2 / sigma_1
    toadd = -mu_1 * sigma_2 / sigma_1 + mu_2

    A_norm_rank_k_temp = A_norm_rank_k_cor[:, toscale]
    A_norm_rank_k_temp = A_norm_rank_k_temp * sigma_1_2[toscale]
    A_norm_rank_k_temp = A_norm_rank_k_temp + toadd[toscale]

    A_norm_rank_k_cor_sc = A_norm_rank_k_cor.copy()
    A_norm_rank_k_cor_sc[:, toscale] = A_norm_rank_k_temp
    A_norm_rank_k_cor_sc[A_norm_rank_k_cor == 0] = 0

    lt0 = A_norm_rank_k_cor_sc < 0
    A_norm_rank_k_cor_sc[lt0] = 0
    print(f"{100 * np.sum(lt0) / (A_norm.shape[0] * A_norm.shape[1]):.2f}% of the values became negative in the scaling process and were set to zero")

    A_norm_rank_k_cor_sc[originally_nonzero & (A_norm_rank_k_cor_sc == 0)] = A_norm[originally_nonzero & (A_norm_rank_k_cor_sc == 0)]

    original_nz = np.sum(A_norm > 0) / (A_norm.shape[0] * A_norm.shape[1])
    completed_nz = np.sum(A_norm_rank_k_cor_sc > 0) / (A_norm.shape[0] * A_norm.shape[1])
    print(f"The matrix went from {100 * original_nz:.2f}% nonzero to {100 * completed_nz:.2f}% nonzero")

    return {'A_norm_rank_k': A_norm_rank_k, 'A_norm_rank_k_cor': A_norm_rank_k_cor, 'A_norm_rank_k_cor_sc': A_norm_rank_k_cor_sc}
