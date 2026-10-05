# Reorganized from spatialVAE/preprocess_boundary.py; see the repository license and attribution notices.
"""
Boundary-aware graph construction for spatial multi-omics data

Key Strategy:
1. Boundary detection uses FUSED features from both omics (more robust)
2. The same boundary_edges (sparse set) is shared by spatial and feature graphs
3. Memory-efficient: O(N*k) instead of O(N²)

Boundary = physical tissue boundary, not modality-specific
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def detect_boundaries_fused(features1, features2, coords, k_neighbors=10,
                            boundary_threshold_factor=2.0, verbose=False):
    """
    Detect tissue boundaries using FUSED features from both omics.

    Memory-efficient: Returns sparse set instead of N×N matrix.

    Parameters
    ----------
    features1 : np.array
        First omics features [n_cells, n_features1]
    features2 : np.array
        Second omics features [n_cells, n_features2]
    coords : np.array
        Spatial coordinates [n_cells, 2]
    k_neighbors : int
        Number of spatial neighbors for boundary detection
    boundary_threshold_factor : float
        Threshold = median_gradient * factor
    verbose : bool
        Print statistics

    Returns
    -------
    boundary_edges : set
        Set of (i, j) tuples where i < j, representing boundary edges
    spatial_indices : np.array
        [n_cells, k+1] neighbor indices (for reuse)
    spatial_distances : np.array
        [n_cells, k+1] neighbor distances (for reuse)
    gradient_threshold : float
        The threshold used
    """
    n_cells = coords.shape[0]

    # 1. Fuse features: standardize each modality, then concatenate
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    feat1_scaled = scaler1.fit_transform(features1)
    feat2_scaled = scaler2.fit_transform(features2)
    feat_fused = np.concatenate([feat1_scaled, feat2_scaled], axis=1)

    if verbose:
        print(f"  Fused features: {features1.shape[1]} + {features2.shape[1]} = {feat_fused.shape[1]} dims")

    # 2. Compute spatial kNN
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(coords)
    spatial_distances, spatial_indices = nbrs.kneighbors(coords)

    # 3. First pass: compute all gradients for adaptive threshold
    all_gradients = []
    for i in range(n_cells):
        for j_idx, j in enumerate(spatial_indices[i, 1:]):  # Skip self
            feature_diff = np.linalg.norm(feat_fused[i] - feat_fused[j])
            spatial_dist = spatial_distances[i, j_idx+1]
            if spatial_dist > 1e-8:
                all_gradients.append(feature_diff / spatial_dist)

    gradient_threshold = np.median(all_gradients) * boundary_threshold_factor

    if verbose:
        print(f"  Gradient threshold: {gradient_threshold:.4f} (median={np.median(all_gradients):.4f}, factor={boundary_threshold_factor})")

    # 4. Second pass: detect boundaries (sparse set)
    boundary_edges = set()

    for i in range(n_cells):
        for j_idx, j in enumerate(spatial_indices[i, 1:]):  # Skip self
            feature_diff = np.linalg.norm(feat_fused[i] - feat_fused[j])
            spatial_dist = spatial_distances[i, j_idx+1]

            if spatial_dist > 1e-8:
                gradient = feature_diff / spatial_dist
            else:
                gradient = feature_diff

            if gradient > gradient_threshold:
                if i < j:
                    boundary_edges.add((i, j))

    if verbose:
        n_total_edges = n_cells * k_neighbors
        print(f"  Detected {len(boundary_edges)}/{n_total_edges} boundary edges ({100*len(boundary_edges)/n_total_edges:.1f}%)")

    return boundary_edges, spatial_indices, spatial_distances, gradient_threshold


def construct_boundary_aware_spatial_graph(
    coords,
    features,
    n_neighbors=6,
    sigma_spatial=1.0,
    gradient_penalty=0.5,
    boundary_edges=None,
    spatial_indices=None,
    spatial_distances=None,
    verbose=False
):
    """
    Construct boundary-aware spatial graph (SPARSE implementation).

    Parameters
    ----------
    coords : np.array
        Spatial coordinates [n_cells, 2]
    features : np.array
        Feature matrix [n_cells, n_features]
    n_neighbors : int
        Number of nearest neighbors
    sigma_spatial : float
        Bandwidth for spatial RBF kernel
    gradient_penalty : float
        Penalty factor for boundary edges (0-1)
    boundary_edges : set, optional
        Pre-computed boundary edges from detect_boundaries_fused()
    spatial_indices : np.array, optional
        Pre-computed neighbor indices
    spatial_distances : np.array, optional
        Pre-computed neighbor distances
    verbose : bool
        Print statistics

    Returns
    -------
    adj_matrix : scipy.sparse.csr_matrix
        Normalized adjacency matrix
    """
    n_cells = coords.shape[0]

    # 1. Build k-NN graph (or use provided)
    if spatial_indices is None or spatial_distances is None:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coords)
        spatial_distances, spatial_indices = nbrs.kneighbors(coords)

    # 2. Normalize features for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # 3. Handle boundary edges
    if boundary_edges is None:
        boundary_edges = set()

    # 4. Construct edge weights (sparse: only on kNN edges)
    row_ind = []
    col_ind = []
    data = []
    n_boundary_in_graph = 0

    #  Record edge weights before attenuation
    boundary_weights_original = []
    non_boundary_weights_original = []

    for i in range(n_cells):
        for j_idx, j in enumerate(spatial_indices[i, 1:]):  # Skip self
            # Spatial weight: RBF kernel
            d_spatial = spatial_distances[i, j_idx+1]
            w_spatial = np.exp(-d_spatial**2 / (2 * sigma_spatial**2))

            # Feature similarity: cosine
            w_feature = np.dot(features_norm[i], features_norm[j])

            # Combined weight
            w_ij = w_spatial * w_feature

            #  Record original weights
            is_boundary = (i, j) in boundary_edges or (j, i) in boundary_edges
            if is_boundary:
                boundary_weights_original.append(w_ij)
            else:
                non_boundary_weights_original.append(w_ij)

            # Apply gradient penalty for boundary edges
            if is_boundary:
                w_ij = w_ij * (1.0 - gradient_penalty)
                n_boundary_in_graph += 1

            row_ind.append(i)
            col_ind.append(j)
            data.append(w_ij)

    # 5. Create symmetric adjacency matrix (use copies to avoid reference issues)
    row = np.array(row_ind, dtype=np.int64)
    col = np.array(col_ind, dtype=np.int64)
    val = np.array(data, dtype=np.float32)

    row_ind = np.concatenate([row, col]).tolist()
    col_ind = np.concatenate([col, row]).tolist()
    data = np.concatenate([val, val]).tolist()

    # 6. Create sparse matrix (self-loops added in normalize_graph)
    adj_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_cells, n_cells))

    # 7. Normalize
    adj_normalized = normalize_graph(adj_matrix, method='symmetric')

    #  Compute diagnostic statistics
    n_total_edges = n_cells * n_neighbors
    boundary_hit_rate = n_boundary_in_graph / n_total_edges if n_total_edges > 0 else 0

    diagnostics = {
        'n_boundary_edges': n_boundary_in_graph,
        'n_total_edges': n_total_edges,
        'boundary_hit_rate': boundary_hit_rate,
        'boundary_weight_mean': np.mean(boundary_weights_original) if boundary_weights_original else 0,
        'non_boundary_weight_mean': np.mean(non_boundary_weights_original) if non_boundary_weights_original else 0,
        'weight_ratio': (np.mean(boundary_weights_original) / np.mean(non_boundary_weights_original)) if non_boundary_weights_original and np.mean(non_boundary_weights_original) > 0 else 0
    }

    if verbose:
        print(f"  Spatial graph: {n_cells} nodes, {n_total_edges} edges")
        print(f"   Flagged edges: {n_boundary_in_graph}/{n_total_edges} ({100*boundary_hit_rate:.2f}%)")
        print(f"   Mean flagged-edge weight: {diagnostics['boundary_weight_mean']:.4f}")
        print(f"   Mean other-edge weight: {diagnostics['non_boundary_weight_mean']:.4f}")
        print(f"   Flagged/other weight ratio: {diagnostics['weight_ratio']:.4f}")
        print(f"   Weight difference after attenuation: {diagnostics['non_boundary_weight_mean'] - diagnostics['boundary_weight_mean']*(1-gradient_penalty):.4f}")

    return adj_normalized, diagnostics


def construct_boundary_aware_feature_graph(
    features,
    n_neighbors=20,
    gradient_penalty=0.3,
    boundary_edges=None,
    verbose=False
):
    """
    Construct boundary-aware feature similarity graph (SPARSE implementation).

    Uses the SAME boundary_edges from spatial detection (unified strategy).

    Parameters
    ----------
    features : np.array
        Feature matrix [n_cells, n_features]
    n_neighbors : int
        Number of nearest neighbors in feature space
    gradient_penalty : float
        Penalty for boundary edges (0-1)
    boundary_edges : set, optional
        Pre-computed boundary edges from detect_boundaries_fused()
    verbose : bool
        Print statistics

    Returns
    -------
    adj_matrix : scipy.sparse.csr_matrix
        Normalized adjacency matrix
    """
    n_cells = features.shape[0]

    # 1. Build k-NN graph in feature space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(features)
    distances, indices = nbrs.kneighbors(features)

    # 2. Normalize features for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # 3. Handle boundary edges
    if boundary_edges is None:
        boundary_edges = set()

    # 4. Construct weighted edges
    row_ind = []
    col_ind = []
    data = []
    n_boundary_in_feature = 0

    #  Record edge weights before attenuation
    boundary_weights_original = []
    non_boundary_weights_original = []

    for i in range(n_cells):
        for j_idx, j in enumerate(indices[i, 1:]):  # Skip self
            # Feature similarity: cosine
            w_ij = np.dot(features_norm[i], features_norm[j])

            #  Record original weights
            is_boundary = (i, j) in boundary_edges or (j, i) in boundary_edges
            if is_boundary:
                boundary_weights_original.append(w_ij)
            else:
                non_boundary_weights_original.append(w_ij)

            # Apply penalty for boundary edges
            if is_boundary:
                w_ij = w_ij * (1.0 - gradient_penalty)
                n_boundary_in_feature += 1

            row_ind.append(i)
            col_ind.append(j)
            data.append(w_ij)

    # 5. Make symmetric
    row = np.array(row_ind, dtype=np.int64)
    col = np.array(col_ind, dtype=np.int64)
    val = np.array(data, dtype=np.float32)

    row_ind = np.concatenate([row, col]).tolist()
    col_ind = np.concatenate([col, row]).tolist()
    data = np.concatenate([val, val]).tolist()

    # 6. Create sparse matrix (self-loops added in normalize_graph)
    adj_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_cells, n_cells))

    # 7. Normalize
    adj_normalized = normalize_graph(adj_matrix, method='symmetric')

    #  Compute diagnostic statistics
    n_total_edges = n_cells * n_neighbors
    boundary_hit_rate = n_boundary_in_feature / n_total_edges if n_total_edges > 0 else 0

    diagnostics = {
        'n_boundary_edges': n_boundary_in_feature,
        'n_total_edges': n_total_edges,
        'boundary_hit_rate': boundary_hit_rate,
        'boundary_weight_mean': np.mean(boundary_weights_original) if boundary_weights_original else 0,
        'non_boundary_weight_mean': np.mean(non_boundary_weights_original) if non_boundary_weights_original else 0,
        'weight_ratio': (np.mean(boundary_weights_original) / np.mean(non_boundary_weights_original)) if non_boundary_weights_original and np.mean(non_boundary_weights_original) > 0 else 0
    }

    if verbose:
        print(f"  Feature graph: {n_cells} nodes, {n_total_edges} edges")
        print(f"   Flagged edges: {n_boundary_in_feature}/{n_total_edges} ({100*boundary_hit_rate:.2f}%)")
        print(f"   Mean flagged-edge weight: {diagnostics['boundary_weight_mean']:.4f}")
        print(f"   Mean other-edge weight: {diagnostics['non_boundary_weight_mean']:.4f}")
        print(f"   Flagged/other weight ratio: {diagnostics['weight_ratio']:.4f}")
        print(f"   Weight difference after attenuation: {diagnostics['non_boundary_weight_mean'] - diagnostics['boundary_weight_mean']*(1-gradient_penalty):.4f}")

    return adj_normalized, diagnostics


def normalize_graph(adj, method='symmetric'):
    """
    Normalize adjacency matrix with self-loops.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix
    method : str
        'symmetric' (D^(-1/2) A D^(-1/2)) or 'random_walk' (D^(-1) A)

    Returns
    -------
    adj_norm : scipy.sparse.csr_matrix
        Normalized adjacency matrix
    """
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


# ============================================================================
# Edge-Preserving Smoothness Loss
# ============================================================================

class EdgePreservingLoss(nn.Module):
    """
    Edge-preserving smoothness loss using Total Variation / Charbonnier penalty.

    Penalizes small differences (smooth within regions) but preserves
    large differences (boundaries).
    """

    def __init__(self, loss_type='charbonnier', epsilon=0.001):
        super(EdgePreservingLoss, self).__init__()
        self.loss_type = loss_type
        self.epsilon = epsilon

    def forward(self, z, adj, edge_weights=None):
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

        # Compute differences
        z_i = z[edges[0]]
        z_j = z[edges[1]]
        diff = z_i - z_j
        diff_norm = torch.norm(diff, p=2, dim=-1)

        # Apply robust penalty
        if self.loss_type == 'l1':
            penalty = torch.abs(diff_norm)
        elif self.loss_type == 'charbonnier':
            penalty = torch.sqrt(diff_norm**2 + self.epsilon**2)
        elif self.loss_type == 'huber':
            delta = 0.1
            mask_small = diff_norm < delta
            penalty = torch.where(
                mask_small,
                0.5 * diff_norm**2 / delta,
                diff_norm - 0.5 * delta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Weight by edge weights
        if edge_weights is not None:
            penalty = penalty * edge_weights
        else:
            penalty = penalty * weights

        return penalty.sum()


class BoundaryAwareRegularizer(nn.Module):
    """Combined regularizer with edge-preserving smoothness."""

    def __init__(self, smooth_weight=1.0, boundary_weight=0.5, loss_type='charbonnier'):
        super(BoundaryAwareRegularizer, self).__init__()
        self.smooth_weight = smooth_weight
        self.boundary_weight = boundary_weight
        self.edge_loss = EdgePreservingLoss(loss_type=loss_type)

    def forward(self, z, adj, boundary_edges=None):
        smooth_loss = self.edge_loss(z, adj)
        boundary_loss = 0.0  # Can be extended
        return self.smooth_weight * smooth_loss + self.boundary_weight * boundary_loss


# ============================================================================
# Main Integration Function
# ============================================================================

def construct_boundary_aware_graphs(
    adata_omics1,
    adata_omics2,
    n_neighbors_spatial=6,
    n_neighbors_feature=20,
    sigma_spatial=1.0,
    gradient_penalty=0.5,
    gradient_penalty_feature=None,
    use_boundary_detection=True,
    boundary_threshold_factor=2.0,
    verbose=True
):
    """
    Construct boundary-aware graphs using UNIFIED boundary detection.

    Strategy:
    1. Detect boundaries using FUSED features from both omics
    2. Share the same boundary_edges for both spatial and feature graphs
    3. Memory-efficient sparse implementation

    Parameters
    ----------
    adata_omics1 : AnnData
        First omics data (must have .obsm['spatial'] and .obsm['feat'])
    adata_omics2 : AnnData
        Second omics data
    n_neighbors_spatial : int
        Neighbors for spatial graph
    n_neighbors_feature : int
        Neighbors for feature graph
    sigma_spatial : float
        Spatial kernel bandwidth
    gradient_penalty : float
        Boundary edge penalty (0-1) for spatial graph
    gradient_penalty_feature : float, optional
        Boundary edge penalty for feature graph. Default: gradient_penalty * 0.5
    use_boundary_detection : bool
        Enable boundary detection
    boundary_threshold_factor : float
        Threshold = median_gradient * factor
    verbose : bool
        Print statistics

    Returns
    -------
    dict with adjacency matrices and boundary_edges
    """
    if gradient_penalty_feature is None:
        gradient_penalty_feature = gradient_penalty * 0.5

    # Extract data
    coords1 = adata_omics1.obsm['spatial']
    coords2 = adata_omics2.obsm['spatial']
    feat1 = adata_omics1.obsm['feat']
    feat2 = adata_omics2.obsm['feat']
    n_cells = coords1.shape[0]

    if verbose:
        print("=" * 60)
        print("Constructing boundary-aware graphs (UNIFIED + FUSED)")
        print("=" * 60)
        print(f"  Spatial neighbors: {n_neighbors_spatial}")
        print(f"  Feature neighbors: {n_neighbors_feature}")
        print(f"  Gradient penalty (spatial): {gradient_penalty}")
        print(f"  Gradient penalty (feature): {gradient_penalty_feature}")

    # Validate consistency
    if coords1.shape != coords2.shape:
        raise ValueError(f"coords1 shape {coords1.shape} != coords2 shape {coords2.shape}")
    if not np.allclose(coords1, coords2):
        import warnings
        warnings.warn("coords1 and coords2 are not identical", UserWarning)

    # =========================================================================
    # Step 1: UNIFIED Boundary Detection using FUSED features
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("Step 1: Boundary Detection (FUSED features)")
        print("-" * 60)

    if use_boundary_detection:
        boundary_edges, spatial_indices, spatial_distances, gradient_threshold = detect_boundaries_fused(
            feat1, feat2, coords1,
            k_neighbors=n_neighbors_spatial,
            boundary_threshold_factor=boundary_threshold_factor,
            verbose=verbose
        )
    else:
        boundary_edges = set()
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_spatial+1).fit(coords1)
        spatial_distances, spatial_indices = nbrs.kneighbors(coords1)
        gradient_threshold = None

    # =========================================================================
    # Step 2: Construct Spatial Graphs (share boundary_edges)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("Step 2: Spatial Graphs")
        print("-" * 60)

    print("\nOmics 1 - Spatial graph:")
    adj_spatial_omics1, diag_spatial_omics1 = construct_boundary_aware_spatial_graph(
        coords1, feat1,
        n_neighbors=n_neighbors_spatial,
        sigma_spatial=sigma_spatial,
        gradient_penalty=gradient_penalty,
        boundary_edges=boundary_edges,
        spatial_indices=spatial_indices,
        spatial_distances=spatial_distances,
        verbose=True
    )

    print("\nOmics 2 - Spatial graph:")
    adj_spatial_omics2, diag_spatial_omics2 = construct_boundary_aware_spatial_graph(
        coords2, feat2,
        n_neighbors=n_neighbors_spatial,
        sigma_spatial=sigma_spatial,
        gradient_penalty=gradient_penalty,
        boundary_edges=boundary_edges,
        spatial_indices=None,  # Recompute for coords2
        spatial_distances=None,
        verbose=True
    )

    # =========================================================================
    # Step 3: Construct Feature Graphs (share SAME boundary_edges)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("Step 3: Feature Graphs (same boundary_edges)")
        print("-" * 60)

    print("\nOmics 1 - Feature graph:")
    adj_feature_omics1, diag_feature_omics1 = construct_boundary_aware_feature_graph(
        feat1,
        n_neighbors=n_neighbors_feature,
        gradient_penalty=gradient_penalty_feature,
        boundary_edges=boundary_edges,
        verbose=True
    )

    print("\nOmics 2 - Feature graph:")
    adj_feature_omics2, diag_feature_omics2 = construct_boundary_aware_feature_graph(
        feat2,
        n_neighbors=n_neighbors_feature,
        gradient_penalty=gradient_penalty_feature,
        boundary_edges=boundary_edges,
        verbose=True
    )

    # =========================================================================
    #  Step 3.5: Print diagnostics
    # =========================================================================
    print("\n" + "=" * 60)
    print(" Graph attenuation diagnostics ")
    print("=" * 60)

    print("\n[Diagnostic 1: flagged-edge fraction]")
    print(f"  Spatial graph Omics1: {diag_spatial_omics1['boundary_hit_rate']*100:.2f}% ({diag_spatial_omics1['n_boundary_edges']}/{diag_spatial_omics1['n_total_edges']})")
    print(f"  Spatial graph Omics2: {diag_spatial_omics2['boundary_hit_rate']*100:.2f}% ({diag_spatial_omics2['n_boundary_edges']}/{diag_spatial_omics2['n_total_edges']})")
    print(f"  Feature graph Omics1: {diag_feature_omics1['boundary_hit_rate']*100:.2f}% ({diag_feature_omics1['n_boundary_edges']}/{diag_feature_omics1['n_total_edges']})")
    print(f"  Feature graph Omics2: {diag_feature_omics2['boundary_hit_rate']*100:.2f}% ({diag_feature_omics2['n_boundary_edges']}/{diag_feature_omics2['n_total_edges']})")

    #  Summarize diagnostics
    avg_spatial_hit = (diag_spatial_omics1['boundary_hit_rate'] + diag_spatial_omics2['boundary_hit_rate']) / 2
    avg_feature_hit = (diag_feature_omics1['boundary_hit_rate'] + diag_feature_omics2['boundary_hit_rate']) / 2

    print(f"\n    Mean flagged-edge fraction: Spatial graph={avg_spatial_hit*100:.2f}%, Feature graph={avg_feature_hit*100:.2f}%")
    if avg_feature_hit < 0.01:
        print("   Very few feature edges are flagged by the boundary threshold.")
    elif avg_feature_hit < 0.05:
        print("    Few feature edges are flagged; inspect the graph settings.")
    else:
        print("   The feature graph includes threshold-flagged edges.")

    print("\n[Diagnostic 2: edge-weight differences]")
    print(f"  Spatial graph Omics1: flagged={diag_spatial_omics1['boundary_weight_mean']:.4f}, other={diag_spatial_omics1['non_boundary_weight_mean']:.4f}, ratio={diag_spatial_omics1['weight_ratio']:.4f}")
    print(f"  Spatial graph Omics2: flagged={diag_spatial_omics2['boundary_weight_mean']:.4f}, other={diag_spatial_omics2['non_boundary_weight_mean']:.4f}, ratio={diag_spatial_omics2['weight_ratio']:.4f}")
    print(f"  Feature graph Omics1: flagged={diag_feature_omics1['boundary_weight_mean']:.4f}, other={diag_feature_omics1['non_boundary_weight_mean']:.4f}, ratio={diag_feature_omics1['weight_ratio']:.4f}")
    print(f"  Feature graph Omics2: flagged={diag_feature_omics2['boundary_weight_mean']:.4f}, other={diag_feature_omics2['non_boundary_weight_mean']:.4f}, ratio={diag_feature_omics2['weight_ratio']:.4f}")

    #  Summarize diagnostics
    avg_weight_ratio = (diag_spatial_omics1['weight_ratio'] + diag_spatial_omics2['weight_ratio'] +
                        diag_feature_omics1['weight_ratio'] + diag_feature_omics2['weight_ratio']) / 4
    print(f"\n    Mean flagged/other weight ratio: {avg_weight_ratio:.4f}")
    if avg_weight_ratio > 0.95:
        print("   Flagged and other edges have similar mean weights.")
    elif avg_weight_ratio > 0.8:
        print("    Mean weights differ little; inspect gradient_penalty.")
    else:
        print("   Flagged and other edges have different mean weights.")

    print("=" * 60)

    # =========================================================================
    # Step 4: Convert to torch sparse tensors
    # =========================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("Step 4: Convert to torch sparse tensors")
        print("-" * 60)

    adj_spatial_omics1_torch = convert_to_torch_sparse(adj_spatial_omics1)
    adj_spatial_omics2_torch = convert_to_torch_sparse(adj_spatial_omics2)
    adj_feature_omics1_torch = convert_to_torch_sparse(adj_feature_omics1)
    adj_feature_omics2_torch = convert_to_torch_sparse(adj_feature_omics2)

    if verbose:
        print("  Done!")
        print("\n" + "=" * 60)
        print("Boundary-aware graph construction completed!")
        print("=" * 60)

    return {
        'adj_spatial_omics1': adj_spatial_omics1_torch,
        'adj_spatial_omics2': adj_spatial_omics2_torch,
        'adj_feature_omics1': adj_feature_omics1_torch,
        'adj_feature_omics2': adj_feature_omics2_torch,
        'boundary_edges': boundary_edges,
        'gradient_threshold': gradient_threshold,
        #  Diagnostic metadata
        'diagnostics': {
            'spatial_omics1': diag_spatial_omics1,
            'spatial_omics2': diag_spatial_omics2,
            'feature_omics1': diag_feature_omics1,
            'feature_omics2': diag_feature_omics2,
            'avg_spatial_hit_rate': avg_spatial_hit,
            'avg_feature_hit_rate': avg_feature_hit,
            'avg_weight_ratio': avg_weight_ratio
        }
    }
