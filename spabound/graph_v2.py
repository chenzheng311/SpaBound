# Reorganized from spatialVAE/preprocess_boundary_v2.py; see the repository license and attribution notices.
"""
Boundary-aware graph construction V2 - Continuous Gradient-based Approach

Key improvements over V1:
1. Spatial graph: Continuous gradient-based penalty (not binary boundary detection)
2. Feature graph: Spatially-constrained feature kNN (not global feature kNN)
3. Both graphs: Per-edge continuous decay based on fused feature gradient

No more "boundary_edges" set - every edge is penalized based on its gradient.
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_fused_features(features1, features2, verbose=False):
    """
    Compute fused features from both modalities.

    Parameters
    ----------
    features1, features2 : np.array
        Features from two modalities

    Returns
    -------
    feat_fused : np.array
        Concatenated and scaled features
    """
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    feat1_scaled = scaler1.fit_transform(features1)
    feat2_scaled = scaler2.fit_transform(features2)
    feat_fused = np.concatenate([feat1_scaled, feat2_scaled], axis=1)

    if verbose:
        print(f"  Fused features: {features1.shape[1]} + {features2.shape[1]} = {feat_fused.shape[1]} dims")

    return feat_fused


def construct_boundary_aware_spatial_graph_v2(
    coords,
    features,
    feat_fused,
    n_neighbors=6,
    sigma_spatial=None,
    beta_spatial=1.0,
    verbose=False
):
    """
    Construct spatial graph with continuous gradient-based penalty.

    Formula:
    w_ij = exp(-d_ij^2 / 2sigma^2) * (1 + cos(f_i, f_j)) / 2 * exp(-beta_s * g_ij)

    where g_ij = ||fused_i - fused_j|| / d_spatial (normalized by median)

    Parameters
    ----------
    coords : np.array
        Spatial coordinates [n_cells, 2]
    features : np.array
        Feature matrix [n_cells, n_features]
    feat_fused : np.array
        Fused features from both modalities [n_cells, n_fused]
    n_neighbors : int
        Number of spatial neighbors
    sigma_spatial : float or None
        Bandwidth for spatial RBF kernel. If None, auto-set to median distance.
    beta_spatial : float
        Gradient penalty strength (higher = stronger boundary preservation)
    verbose : bool
        Print statistics

    Returns
    -------
    adj_normalized : scipy.sparse.csr_matrix
    diagnostics : dict
    edge_gradients : dict
        Gradients for each edge (for use in edge-preserving loss)
    """
    n_cells = coords.shape[0]

    # 1. Build spatial kNN
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coords)
    spatial_distances, spatial_indices = nbrs.kneighbors(coords)

    #  Fix 2: Auto-adaptive sigma_spatial
    if sigma_spatial is None:
        sigma_spatial = np.median(spatial_distances[:, 1:])  # Exclude self (distance 0)
        if verbose:
            print(f"   Adaptive sigma_spatial: {sigma_spatial:.4f}")

    # 2. Normalize features for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    #  Fix 3: Removed unused feat_fused_norm

    # 3. First pass: compute all gradients for normalization
    all_gradients = []
    edge_info = []  # Store (i, j, d_spatial, cos_sim, gradient)

    for i in range(n_cells):
        for j_idx, j in enumerate(spatial_indices[i, 1:]):  # Skip self
            d_spatial = spatial_distances[i, j_idx+1]
            cos_sim = np.dot(features_norm[i], features_norm[j])

            # Gradient = fused feature difference / spatial distance
            if d_spatial > 1e-8:
                gradient = np.linalg.norm(feat_fused[i] - feat_fused[j]) / d_spatial
            else:
                gradient = np.linalg.norm(feat_fused[i] - feat_fused[j])

            all_gradients.append(gradient)
            edge_info.append((i, j, d_spatial, cos_sim, gradient))

    # 4. Normalize gradients by median
    gradient_median = np.median(all_gradients) + 1e-8
    normalized_gradients = np.array(all_gradients) / gradient_median

    # 5. Compute edge weights with continuous penalty
    row_ind = []
    col_ind = []
    data = []
    edge_grad_list = []  #  Fix 8: Store gradients for each edge

    # For diagnostics
    weights_no_penalty = []
    weights_with_penalty = []
    gradient_values = []

    for idx, (i, j, d_spatial, cos_sim, gradient) in enumerate(edge_info):
        g_hat = normalized_gradients[idx]

        # Spatial RBF weight
        w_spatial = np.exp(-d_spatial**2 / (2 * sigma_spatial**2))

        # Feature similarity weight (cosine, normalized to [0, 1])
        w_feature = (1 + cos_sim) / 2

        # Base weight
        w_base = w_spatial * w_feature

        # Continuous gradient penalty
        w_penalty = np.exp(-beta_spatial * g_hat)

        # Final weight
        w_ij = w_base * w_penalty

        row_ind.append(i)
        col_ind.append(j)
        data.append(w_ij)
        edge_grad_list.append(g_hat)  #  Fix 8: Store gradient

        # Diagnostics
        weights_no_penalty.append(w_base)
        weights_with_penalty.append(w_ij)
        gradient_values.append(g_hat)

    # 6. Make symmetric
    n_edges_directed = len(row_ind)  #  Fix 1: Accurate edge count
    row = np.array(row_ind, dtype=np.int64)
    col = np.array(col_ind, dtype=np.int64)
    val = np.array(data, dtype=np.float32)
    edge_grad_arr = np.array(edge_grad_list)

    #  Fix 8: Symmetrize gradients too
    row_ind = np.concatenate([row, col]).tolist()
    col_ind = np.concatenate([col, row]).tolist()
    data = np.concatenate([val, val]).tolist()
    edge_gradients_sym = np.concatenate([edge_grad_arr, edge_grad_arr])

    # 7. Create sparse matrix
    adj_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_cells, n_cells))
    n_edges_undirected = adj_matrix.nnz // 2  #  Fix 1: More accurate

    # 8. Normalize
    adj_normalized = normalize_graph(adj_matrix, method='symmetric')

    # 9. Diagnostics
    weights_no_penalty = np.array(weights_no_penalty)
    weights_with_penalty = np.array(weights_with_penalty)
    gradient_values = np.array(gradient_values)

    #  Fix 6: Multiple threshold levels for gradient classification
    gradient_80pct = np.quantile(gradient_values, 0.8)
    gradient_90pct = np.quantile(gradient_values, 0.9)

    above_median_mask = gradient_values > 1.0
    very_high_gradient_mask = gradient_values > gradient_80pct
    extreme_gradient_mask = gradient_values > gradient_90pct

    diagnostics = {
        'n_edges_directed': n_edges_directed,
        'n_edges_undirected': n_edges_undirected,
        'gradient_median': gradient_median,
        'gradient_80pct': gradient_80pct,
        'gradient_90pct': gradient_90pct,
        'sigma_spatial': sigma_spatial,
        'beta': beta_spatial,
        'weight_mean_no_penalty': np.mean(weights_no_penalty),
        'weight_mean_with_penalty': np.mean(weights_with_penalty),
        'weight_reduction_ratio': 1 - np.mean(weights_with_penalty) / (np.mean(weights_no_penalty) + 1e-8),
        'above_median_ratio': np.mean(above_median_mask),
        'very_high_gradient_ratio': np.mean(very_high_gradient_mask),
        'extreme_gradient_ratio': np.mean(extreme_gradient_mask),
        'above_median_weight': np.mean(weights_with_penalty[above_median_mask]) if np.any(above_median_mask) else 0,
        'below_median_weight': np.mean(weights_with_penalty[~above_median_mask]) if np.any(~above_median_mask) else 0,
        'very_high_grad_weight': np.mean(weights_with_penalty[very_high_gradient_mask]) if np.any(very_high_gradient_mask) else 0,
        'normal_grad_weight': np.mean(weights_with_penalty[~very_high_gradient_mask]) if np.any(~very_high_gradient_mask) else 0,
    }

    #  Fix 8: Create edge_gradients dict for use in loss
    edge_gradients = {
        'row': np.array(row_ind, dtype=np.int64),
        'col': np.array(col_ind, dtype=np.int64),
        'gradients': edge_gradients_sym.astype(np.float32),
        'gradient_80pct': gradient_80pct,
        'gradient_90pct': gradient_90pct,
    }

    if verbose:
        print(f"  Spatial graph V2: {n_cells} nodes, {n_edges_directed} directed edges ({n_edges_undirected} undirected)")
        print(f"   sigma_spatial: {sigma_spatial:.4f}, beta: {beta_spatial:.4f}")
        print(f"   Median gradient: {gradient_median:.4f}, 80th percentile: {gradient_80pct:.4f}, 90th percentile: {gradient_90pct:.4f}")
        print(f"   Edges above median gradient: {100*diagnostics['above_median_ratio']:.1f}%")
        print(f"   Edges above 80th percentile: {100*diagnostics['very_high_gradient_ratio']:.1f}%")
        print(f"   Edges above 90th percentile: {100*diagnostics['extreme_gradient_ratio']:.1f}%")
        print(f"   Mean weight before attenuation: {diagnostics['weight_mean_no_penalty']:.4f}")
        print(f"   Mean weight after attenuation: {diagnostics['weight_mean_with_penalty']:.4f}")
        print(f"   Overall weight reduction: {100*diagnostics['weight_reduction_ratio']:.2f}%")
        high_low_ratio = diagnostics['above_median_weight'] / (diagnostics['below_median_weight'] + 1e-8)
        print(f"   High/low gradient weight ratio: {high_low_ratio:.4f}")

    return adj_normalized, diagnostics, edge_gradients


def construct_spatially_constrained_feature_graph_v2(
    coords,
    features,
    feat_fused,
    spatial_candidates_k=25,
    n_feature_neighbors=10,
    beta_feature=1.0,
    spatial_radius=None,
    verbose=False
):
    """
    Construct feature graph with SPATIAL CONSTRAINTS.

    Key insight: Feature neighbors should be selected from spatial candidates first,
    not from global feature space.

    For each cell i:
    1. Find spatial candidates (e.g., 25 nearest spatial neighbors)
    2. From these candidates, select top-k by feature similarity
    3. Apply continuous gradient penalty to each edge

    Formula:
    w_ij = (1 + cos(f_i, f_j)) / 2 * exp(-beta_f * g_ij_fused)

    where g_ij_fused = ||fused_i - fused_j|| (normalized by median)

    Parameters
    ----------
    coords : np.array
        Spatial coordinates [n_cells, 2]
    features : np.array
        Feature matrix [n_cells, n_features]
    feat_fused : np.array
        Fused features from both modalities [n_cells, n_fused]
    spatial_candidates_k : int
        Number of spatial candidates to consider (default: 25)
    n_feature_neighbors : int
        Number of feature neighbors to select (default: 10)
    beta_feature : float
        Gradient penalty strength for feature graph
    spatial_radius : float, optional
        If provided, use spatial radius instead of k candidates
    verbose : bool
        Print statistics

    Returns
    -------
    adj_normalized : scipy.sparse.csr_matrix
    diagnostics : dict
    edge_gradients : dict
        Gradients for each edge (for use in edge-preserving loss)
    """
    n_cells = coords.shape[0]

    # 1. Find spatial candidates (more than feature neighbors)
    if spatial_radius is not None:
        # Use radius-based candidates
        nbrs = NearestNeighbors(radius=spatial_radius).fit(coords)
        spatial_candidates = nbrs.radius_neighbors(coords, return_distance=False)
    else:
        # Use k-based candidates
        nbrs = NearestNeighbors(n_neighbors=spatial_candidates_k+1).fit(coords)
        _, spatial_indices = nbrs.kneighbors(coords)
        spatial_candidates = [indices[1:] for indices in spatial_indices]  # Remove self

    # 2. Normalize features for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # 3. Build feature edges from spatial candidates
    #  Fix 5: Use unified edge_info structure
    edge_info = []  # (i, j, cos_sim, fused_diff)

    for i in range(n_cells):
        candidates = spatial_candidates[i]

        if len(candidates) == 0:
            continue

        # Compute cosine similarity with all candidates
        cos_sims = np.array([np.dot(features_norm[i], features_norm[j]) for j in candidates])

        #  Fix 4: Clearer top-k selection (descending order)
        k = min(n_feature_neighbors, len(candidates))
        topk_local_idx = np.argsort(-cos_sims)[:k]  # Descending: highest first

        for local_idx in topk_local_idx:
            j = candidates[local_idx]
            cos_sim = cos_sims[local_idx]
            fused_diff = np.linalg.norm(feat_fused[i] - feat_fused[j])
            edge_info.append((i, j, cos_sim, fused_diff))

    # 4. Normalize fused differences by median
    all_fused_diffs = np.array([e[3] for e in edge_info])
    all_cosine_sims = np.array([e[2] for e in edge_info])

    fused_diff_median = np.median(all_fused_diffs) + 1e-8
    normalized_diffs = all_fused_diffs / fused_diff_median

    # 5. Compute final weights with continuous penalty
    row_ind = []
    col_ind = []
    data = []
    edge_grad_list = []  #  Fix 8: Store gradients

    weights_no_penalty = []
    weights_with_penalty = []

    for idx, (i, j, cos_sim, fused_diff) in enumerate(edge_info):
        g_hat = normalized_diffs[idx]

        # Base weight (cosine similarity, normalized to [0, 1])
        w_base = (1 + cos_sim) / 2

        # Continuous gradient penalty
        w_penalty = np.exp(-beta_feature * g_hat)

        # Final weight
        w_ij = w_base * w_penalty

        row_ind.append(i)
        col_ind.append(j)
        data.append(w_ij)
        edge_grad_list.append(g_hat)

        weights_no_penalty.append(w_base)
        weights_with_penalty.append(w_ij)

    # 6. Make symmetric
    n_edges_directed = len(row_ind)  #  Fix 1: Accurate edge count
    row = np.array(row_ind, dtype=np.int64)
    col = np.array(col_ind, dtype=np.int64)
    val = np.array(data, dtype=np.float32)
    edge_grad_arr = np.array(edge_grad_list)

    #  Fix 8: Symmetrize gradients too
    row_ind = np.concatenate([row, col]).tolist()
    col_ind = np.concatenate([col, row]).tolist()
    data = np.concatenate([val, val]).tolist()
    edge_gradients_sym = np.concatenate([edge_grad_arr, edge_grad_arr])

    # 7. Create sparse matrix
    adj_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_cells, n_cells))
    n_edges_undirected = adj_matrix.nnz // 2

    # 8. Normalize
    adj_normalized = normalize_graph(adj_matrix, method='symmetric')

    # 9. Diagnostics
    weights_no_penalty = np.array(weights_no_penalty)
    weights_with_penalty = np.array(weights_with_penalty)

    #  Fix 6: Multiple threshold levels
    gradient_80pct = np.quantile(normalized_diffs, 0.8)
    gradient_90pct = np.quantile(normalized_diffs, 0.9)

    above_median_mask = normalized_diffs > 1.0
    very_high_gradient_mask = normalized_diffs > gradient_80pct

    diagnostics = {
        'n_edges_directed': n_edges_directed,
        'n_edges_undirected': n_edges_undirected,
        'spatial_candidates_k': spatial_candidates_k,
        'n_feature_neighbors': n_feature_neighbors,
        'fused_diff_median': fused_diff_median,
        'gradient_80pct': gradient_80pct,
        'gradient_90pct': gradient_90pct,
        'beta': beta_feature,
        'weight_mean_no_penalty': np.mean(weights_no_penalty),
        'weight_mean_with_penalty': np.mean(weights_with_penalty),
        'weight_reduction_ratio': 1 - np.mean(weights_with_penalty) / (np.mean(weights_no_penalty) + 1e-8),
        'above_median_ratio': np.mean(above_median_mask),
        'very_high_gradient_ratio': np.mean(very_high_gradient_mask),
        'above_median_weight': np.mean(weights_with_penalty[above_median_mask]) if np.any(above_median_mask) else 0,
        'below_median_weight': np.mean(weights_with_penalty[~above_median_mask]) if np.any(~above_median_mask) else 0,
        'very_high_grad_weight': np.mean(weights_with_penalty[very_high_gradient_mask]) if np.any(very_high_gradient_mask) else 0,
        'normal_grad_weight': np.mean(weights_with_penalty[~very_high_gradient_mask]) if np.any(~very_high_gradient_mask) else 0,
    }

    #  Fix 8: Create edge_gradients dict
    edge_gradients = {
        'row': np.array(row_ind, dtype=np.int64),
        'col': np.array(col_ind, dtype=np.int64),
        'gradients': edge_gradients_sym.astype(np.float32),
        'gradient_80pct': gradient_80pct,
        'gradient_90pct': gradient_90pct,
    }

    if verbose:
        print(f"  Feature graph V2 (spatial-constrained): {n_cells} nodes, {n_edges_directed} directed edges ({n_edges_undirected} undirected)")
        print(f"   Spatial candidates: {spatial_candidates_k}, Feature neighbors: {n_feature_neighbors}")
        print(f"   Median fused discrepancy: {fused_diff_median:.4f}, beta: {beta_feature:.4f}")
        print(f"   Edges above median gradient: {100*diagnostics['above_median_ratio']:.1f}%")
        print(f"   Edges above 80th percentile: {100*diagnostics['very_high_gradient_ratio']:.1f}%")
        print(f"   Mean weight before attenuation: {diagnostics['weight_mean_no_penalty']:.4f}")
        print(f"   Mean weight after attenuation: {diagnostics['weight_mean_with_penalty']:.4f}")
        print(f"   Overall weight reduction: {100*diagnostics['weight_reduction_ratio']:.2f}%")
        high_low_ratio = diagnostics['above_median_weight'] / (diagnostics['below_median_weight'] + 1e-8)
        print(f"   High/low gradient weight ratio: {high_low_ratio:.4f}")

    return adj_normalized, diagnostics, edge_gradients


def normalize_graph(adj, method='symmetric'):
    """Normalize adjacency matrix with self-loops."""
    from scipy.sparse import eye

    # Add self-loops
    adj = adj + eye(adj.shape[0], format='csr')

    if method == 'symmetric':
        degree = np.array(adj.sum(1)).flatten()
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        degree_mat_inv_sqrt = csr_matrix(np.diag(degree_inv_sqrt))
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
    elif method == 'random_walk':
        degree = np.array(adj.sum(1)).flatten()
        degree_inv = np.power(degree, -1)
        degree_inv[np.isinf(degree_inv)] = 0
        degree_mat_inv = csr_matrix(np.diag(degree_inv))
        adj_norm = degree_mat_inv @ adj
    else:
        adj_norm = adj

    return adj_norm


def convert_to_torch_sparse(adj):
    """Convert scipy sparse matrix to torch sparse tensor"""
    adj = adj.tocoo()
    indices = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj.data.astype(np.float32))
    shape = torch.Size(adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def construct_boundary_aware_graphs_v2(
    adata_omics1,
    adata_omics2,
    # Spatial graph parameters
    n_neighbors_spatial=6,
    sigma_spatial=None,
    beta_spatial=1.0,
    # Feature graph parameters
    spatial_candidates_k=25,
    n_neighbors_feature=10,
    beta_feature=1.0,
    verbose=True
):
    """
    Construct boundary-aware graphs V2 with continuous gradient-based approach.

    Key improvements:
    1. Spatial graph: Continuous penalty based on normalized gradient
    2. Feature graph: Spatially-constrained feature kNN + continuous penalty
    3. No more discrete boundary_edges set
    4.  Fix 8: Returns edge_gradients for use in edge-preserving loss

    Parameters
    ----------
    adata_omics1, adata_omics2 : AnnData
        Omics data with .obsm['spatial'] and .obsm['feat']
    n_neighbors_spatial : int
        Spatial neighbors for spatial graph
    sigma_spatial : float or None
        Spatial RBF bandwidth. If None, auto-set to median distance.  Fix 2
    beta_spatial : float
        Gradient penalty strength for spatial graph (higher = stronger boundary)
    spatial_candidates_k : int
        Spatial candidates for feature graph (should >= n_neighbors_feature)
    n_neighbors_feature : int
        Feature neighbors to select from spatial candidates
    beta_feature : float
        Gradient penalty strength for feature graph
    verbose : bool
        Print statistics

    Returns
    -------
    dict with adjacency matrices, diagnostics, and edge_gradients
    """
    # Extract data
    coords1 = adata_omics1.obsm['spatial']
    coords2 = adata_omics2.obsm['spatial']
    feat1 = adata_omics1.obsm['feat']
    feat2 = adata_omics2.obsm['feat']
    n_cells = coords1.shape[0]

    if verbose:
        print("=" * 70)
        print("Boundary-Aware Graph Construction V2 (Continuous Gradient-Based)")
        print("=" * 70)
        print(f"  N cells: {n_cells}")
        print(f"  Spatial graph: k={n_neighbors_spatial}, sigma={'auto' if sigma_spatial is None else sigma_spatial}, beta={beta_spatial}")
        print(f"  Feature graph: candidates={spatial_candidates_k}, k={n_neighbors_feature}, beta={beta_feature}")

    # =========================================================================
    # Step 1: Compute fused features
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("Step 1: Compute Fused Features")
        print("-" * 70)

    feat_fused = compute_fused_features(feat1, feat2, verbose=verbose)

    # =========================================================================
    # Step 2: Construct Spatial Graphs
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("Step 2: Spatial Graphs (Continuous Gradient Penalty)")
        print("-" * 70)

    print("\nOmics 1 - Spatial graph:")
    adj_spatial_omics1, diag_spatial_omics1, grad_spatial_omics1 = construct_boundary_aware_spatial_graph_v2(
        coords1, feat1, feat_fused,
        n_neighbors=n_neighbors_spatial,
        sigma_spatial=sigma_spatial,
        beta_spatial=beta_spatial,
        verbose=True
    )

    print("\nOmics 2 - Spatial graph:")
    adj_spatial_omics2, diag_spatial_omics2, grad_spatial_omics2 = construct_boundary_aware_spatial_graph_v2(
        coords2, feat2, feat_fused,
        n_neighbors=n_neighbors_spatial,
        sigma_spatial=sigma_spatial,
        beta_spatial=beta_spatial,
        verbose=True
    )

    # =========================================================================
    # Step 3: Construct Feature Graphs (Spatially-Constrained)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("Step 3: Feature Graphs (Spatially-Constrained + Continuous Penalty)")
        print("-" * 70)

    print("\nOmics 1 - Feature graph:")
    adj_feature_omics1, diag_feature_omics1, grad_feature_omics1 = construct_spatially_constrained_feature_graph_v2(
        coords1, feat1, feat_fused,
        spatial_candidates_k=spatial_candidates_k,
        n_feature_neighbors=n_neighbors_feature,
        beta_feature=beta_feature,
        verbose=True
    )

    print("\nOmics 2 - Feature graph:")
    adj_feature_omics2, diag_feature_omics2, grad_feature_omics2 = construct_spatially_constrained_feature_graph_v2(
        coords2, feat2, feat_fused,
        spatial_candidates_k=spatial_candidates_k,
        n_feature_neighbors=n_neighbors_feature,
        beta_feature=beta_feature,
        verbose=True
    )

    # =========================================================================
    # Step 4: Summary Diagnostics
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Continuous graph attenuation diagnostics ")
    print("=" * 70)

    print("\n[Diagnostic: High/low gradient weight ratio]")
    #  Fix 7: More cautious interpretation
    print("  (Smaller ratios indicate stronger attenuation of high-gradient edges; inspect alongside downstream results)")

    spatial_ratio_1 = diag_spatial_omics1['above_median_weight'] / (diag_spatial_omics1['below_median_weight'] + 1e-8)
    spatial_ratio_2 = diag_spatial_omics2['above_median_weight'] / (diag_spatial_omics2['below_median_weight'] + 1e-8)
    feature_ratio_1 = diag_feature_omics1['above_median_weight'] / (diag_feature_omics1['below_median_weight'] + 1e-8)
    feature_ratio_2 = diag_feature_omics2['above_median_weight'] / (diag_feature_omics2['below_median_weight'] + 1e-8)

    print(f"  Spatial graph Omics1: {spatial_ratio_1:.4f}")
    print(f"  Spatial graph Omics2: {spatial_ratio_2:.4f}")
    print(f"  Feature graph Omics1: {feature_ratio_1:.4f}")
    print(f"  Feature graph Omics2: {feature_ratio_2:.4f}")

    avg_ratio = (spatial_ratio_1 + spatial_ratio_2 + feature_ratio_1 + feature_ratio_2) / 4
    print(f"\n    Mean high/low gradient weight ratio: {avg_ratio:.4f}")

    #  Fix 7: More cautious interpretation without hard thresholds
    if avg_ratio < 0.3:
        print("   Strong attenuation of high-gradient edges")
    elif avg_ratio < 0.5:
        print("   High-gradient edges receive attenuated weights")
    elif avg_ratio < 0.7:
        print("   Moderate attenuation; inspect beta settings")
    else:
        print("   Weak attenuation; inspect beta_spatial and beta_feature")

    print("\n[Edge counts]")  #  Fix 1: Accurate edge counts
    print(f"  Spatial graph Omics1: {diag_spatial_omics1['n_edges_directed']} directed ({diag_spatial_omics1['n_edges_undirected']} undirected)")
    print(f"  Spatial graph Omics2: {diag_spatial_omics2['n_edges_directed']} directed ({diag_spatial_omics2['n_edges_undirected']} undirected)")
    print(f"  Feature graph Omics1: {diag_feature_omics1['n_edges_directed']} directed ({diag_feature_omics1['n_edges_undirected']} undirected)")
    print(f"  Feature graph Omics2: {diag_feature_omics2['n_edges_directed']} directed ({diag_feature_omics2['n_edges_undirected']} undirected)")

    print("\n[Weight reductions]")
    print(f"  Spatial graph Omics1: {100*diag_spatial_omics1['weight_reduction_ratio']:.2f}%")
    print(f"  Spatial graph Omics2: {100*diag_spatial_omics2['weight_reduction_ratio']:.2f}%")
    print(f"  Feature graph Omics1: {100*diag_feature_omics1['weight_reduction_ratio']:.2f}%")
    print(f"  Feature graph Omics2: {100*diag_feature_omics2['weight_reduction_ratio']:.2f}%")

    #  Fix 6: Show very high gradient statistics
    print("\n[Edges above 80th percentile]")
    print(f"  Spatial graph Omics1: {100*diag_spatial_omics1['very_high_gradient_ratio']:.1f}%, weight={diag_spatial_omics1['very_high_grad_weight']:.4f}")
    print(f"  Spatial graph Omics2: {100*diag_spatial_omics2['very_high_gradient_ratio']:.1f}%, weight={diag_spatial_omics2['very_high_grad_weight']:.4f}")
    print(f"  Feature graph Omics1: {100*diag_feature_omics1['very_high_gradient_ratio']:.1f}%, weight={diag_feature_omics1['very_high_grad_weight']:.4f}")
    print(f"  Feature graph Omics2: {100*diag_feature_omics2['very_high_gradient_ratio']:.1f}%, weight={diag_feature_omics2['very_high_grad_weight']:.4f}")

    print("=" * 70)

    # =========================================================================
    # Step 5: Convert to torch sparse tensors
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("Step 5: Convert to torch sparse tensors")
        print("-" * 70)

    adj_spatial_omics1_torch = convert_to_torch_sparse(adj_spatial_omics1)
    adj_spatial_omics2_torch = convert_to_torch_sparse(adj_spatial_omics2)
    adj_feature_omics1_torch = convert_to_torch_sparse(adj_feature_omics1)
    adj_feature_omics2_torch = convert_to_torch_sparse(adj_feature_omics2)

    if verbose:
        print("  Done!")

    return {
        'adj_spatial_omics1': adj_spatial_omics1_torch,
        'adj_spatial_omics2': adj_spatial_omics2_torch,
        'adj_feature_omics1': adj_feature_omics1_torch,
        'adj_feature_omics2': adj_feature_omics2_torch,
        'feat_fused': feat_fused,
        'diagnostics': {
            'spatial_omics1': diag_spatial_omics1,
            'spatial_omics2': diag_spatial_omics2,
            'feature_omics1': diag_feature_omics1,
            'feature_omics2': diag_feature_omics2,
            'avg_high_low_ratio': avg_ratio
        },
        #  Fix 8: Include edge gradients for use in edge-preserving loss
        'edge_gradients': {
            'spatial_omics1': grad_spatial_omics1,
            'spatial_omics2': grad_spatial_omics2,
            'feature_omics1': grad_feature_omics1,
            'feature_omics2': grad_feature_omics2,
        }
    }


# ============================================================================
# Edge-Preserving Loss V2 - Only smooth on low-gradient edges
# ============================================================================

class EdgePreservingLossV2(torch.nn.Module):
    """
    Edge-preserving smoothness loss that only smooths low-gradient edges.

     Optimized version with caching to avoid Python loops in forward().

    High-gradient edges (boundaries) are excluded from smoothness loss.
    This prevents the loss from "smoothing away" tissue boundaries.

    Usage:
        # During graph construction
        result = construct_boundary_aware_graphs_v2(...)
        edge_grad = result['edge_gradients']['spatial_omics1']

        # During training
        loss_fn = EdgePreservingLossV2(gradient_threshold=edge_grad['gradient_80pct'])
        loss = loss_fn(z, adj, edge_grad['gradients'])
    """

    def __init__(self, epsilon=0.001, gradient_threshold=1.0):
        """
        Parameters
        ----------
        epsilon : float
            Small constant for Charbonnier penalty (default: 0.001)
        gradient_threshold : float
            Edges with normalized gradient >= threshold are excluded from smoothing.
            - 1.0 = exclude above-median edges
            - Use gradient_80pct from edge_gradients dict to exclude top 20%
            - Use gradient_90pct from edge_gradients dict to exclude top 10%
        """
        super(EdgePreservingLossV2, self).__init__()
        self.epsilon = epsilon
        self.gradient_threshold = gradient_threshold
        #  Cache for preprocessed gradient tensor
        self._cached_grad_key = None
        self._cached_grad_tensor = None

    def _preprocess_gradients(self, edge_gradients_dict, adj, device):
        """
        Preprocess edge gradients into a tensor matching adjacency matrix order.
        Uses caching to avoid recomputation.

         This is called once per adjacency matrix, not every forward pass.
        """
        if adj.is_sparse:
            adj_coal = adj.coalesce()
            edges = adj_coal.indices()
        else:
            edges = torch.nonzero(adj, as_tuple=False).t()

        # Filter out self-loops
        mask = edges[0] != edges[1]
        edges = edges[:, mask]

        # Create cache key based on adjacency matrix id
        cache_key = id(adj)

        # Check cache
        if self._cached_grad_key == cache_key and self._cached_grad_tensor is not None:
            return self._cached_grad_tensor, edges

        if isinstance(edge_gradients_dict, dict):
            # Build gradient tensor from dict using vectorized operations
            grad_row = edge_gradients_dict['row']
            grad_col = edge_gradients_dict['col']
            grad_vals = edge_gradients_dict['gradients']

            n_edges = edges.shape[1]
            edge_grads = torch.zeros(n_edges, dtype=torch.float32, device=device)

            # Build lookup dictionary (still need this for matching)
            grad_map = {}
            for idx in range(len(grad_row)):
                key = (int(grad_row[idx]), int(grad_col[idx]))
                grad_map[key] = float(grad_vals[idx])

            # Match edges to gradients
            for idx in range(n_edges):
                i, j = int(edges[0, idx]), int(edges[1, idx])
                if (i, j) in grad_map:
                    edge_grads[idx] = grad_map[(i, j)]
                elif (j, i) in grad_map:
                    edge_grads[idx] = grad_map[(j, i)]

            # Cache the result
            self._cached_grad_key = cache_key
            self._cached_grad_tensor = edge_grads

            return edge_grads, edges
        else:
            # Already a tensor
            return edge_gradients_dict, edges

    def forward(self, z, adj, edge_gradients_dict=None):
        """
        Parameters
        ----------
        z : torch.Tensor
            Latent representation [n_cells, n_features]
        adj : torch.Tensor
            Adjacency matrix (sparse or dense)
        edge_gradients_dict : dict or torch.Tensor
            If dict: {'row': ..., 'col': ..., 'gradients': ...} from graph construction
            If torch.Tensor: gradient values matching adj edge order
            If None: use adjacency weights as proxy (less accurate)
        """
        # Get edges from sparse adjacency
        if adj.is_sparse:
            adj_coal = adj.coalesce()
            edges = adj_coal.indices()
            weights = adj_coal.values()
        else:
            edges = torch.nonzero(adj, as_tuple=False).t()
            weights = adj[edges[0], edges[1]]

        # Filter out self-loops
        mask = edges[0] != edges[1]
        edges = edges[:, mask]
        weights = weights[mask]

        # Handle edge gradients
        if edge_gradients_dict is not None:
            if isinstance(edge_gradients_dict, dict):
                #  Use cached preprocessing
                edge_gradients_tensor, _ = self._preprocess_gradients(edge_gradients_dict, adj, z.device)
            else:
                # Assume it's already a tensor
                edge_gradients_tensor = edge_gradients_dict
                if edge_gradients_tensor.shape[0] > mask.shape[0]:
                    edge_gradients_tensor = edge_gradients_tensor[mask]

            # Only smooth low-gradient edges
            low_grad_mask = edge_gradients_tensor < self.gradient_threshold
            edges = edges[:, low_grad_mask]
            weights = weights[low_grad_mask]

        # Compute differences
        z_i = z[edges[0]]
        z_j = z[edges[1]]
        diff = z_i - z_j
        diff_norm = torch.norm(diff, p=2, dim=-1)

        # Charbonnier penalty (robust to outliers)
        penalty = torch.sqrt(diff_norm**2 + self.epsilon**2)

        # Weight by edge weights (already includes gradient penalty from graph construction)
        penalty = penalty * weights

        return penalty.mean()  # Use mean for stability across different edge counts
