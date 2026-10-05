"""SpaBound: boundary-aware representation learning for aligned spatial omics."""

from .trainer import SpaBound
from .model import SpaBoundModel, SpaBoundPrivateReconModel
from .graph import construct_boundary_aware_graphs
from .graph_v2 import construct_boundary_aware_graphs_v2
from .preprocessing import (
    adjacent_matrix_preprocessing,
    clr_normalize_each_cell,
    construct_neighbor_graph,
    fix_seed,
    lsi,
    pca,
    tfidf,
)
from .utils import clustering, mclust_R, plot_weight_value, search_res
from .losses import EdgePreservingLoss, KLAnnealer

__version__ = "0.1.0"

__all__ = [
    "SpaBound",
    "SpaBoundModel",
    "SpaBoundPrivateReconModel",
    "construct_boundary_aware_graphs",
    "construct_boundary_aware_graphs_v2",
    "adjacent_matrix_preprocessing",
    "clr_normalize_each_cell",
    "construct_neighbor_graph",
    "fix_seed",
    "lsi",
    "pca",
    "tfidf",
    "clustering",
    "mclust_R",
    "plot_weight_value",
    "search_res",
    "EdgePreservingLoss",
    "KLAnnealer",
]
