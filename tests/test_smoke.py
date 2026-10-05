"""Small installation tests; no benchmark data, GPU, or R is required."""
import numpy as np
import pytest
import torch
from anndata import AnnData

from spabound import SpaBound, construct_boundary_aware_graphs_v2


@pytest.fixture
def paired_data():
    rng = np.random.default_rng(17)
    n = 36
    coordinates = np.column_stack(np.unravel_index(np.arange(n), (6, 6))).astype(float)
    domains = (coordinates[:, 0] >= 3).astype(float)
    data = []
    for width in (8, 6):
        features = (rng.normal(size=(n, width)) + 2 * domains[:, None]).astype(np.float32)
        adata = AnnData(features.copy())
        adata.obsm['feat'] = features
        adata.obsm['spatial'] = coordinates.copy()
        data.append(adata)
    return data


@pytest.mark.parametrize('use_appnp', [True, False])
def test_graph_training_and_export(paired_data, use_appnp):
    torch.set_num_threads(1)
    first, second = paired_data
    graphs = construct_boundary_aware_graphs_v2(
        first, second,
        n_neighbors_spatial=4,
        spatial_candidates_k=8,
        n_neighbors_feature=3,
        beta_spatial=0.8,
        beta_feature=1.2,
        verbose=False,
    )
    for key in ('adj_spatial_omics1', 'adj_spatial_omics2',
                'adj_feature_omics1', 'adj_feature_omics2'):
        matrix = graphs[key].to_dense()
        assert matrix.shape == (36, 36)
        assert torch.isfinite(matrix).all()
        assert (matrix >= 0).all()
        torch.testing.assert_close(matrix, matrix.T)

    trainer = SpaBound(
        {'adata_omics1': first, 'adata_omics2': second, **graphs},
        device=torch.device('cpu'), random_seed=23,
        epochs=2, dim_output=12, z_dim=8,
        n_latent_shared=4, n_latent_private=2,
        appnp_hidden=16, appnp_layers=2,
        use_appnp=use_appnp, use_edge_preserving_loss=True,
    )
    output = trainer.train()
    embedding = output['SpaBound']
    assert embedding.shape == (36, 4)
    assert np.isfinite(embedding).all()
    np.testing.assert_allclose(np.linalg.norm(embedding, axis=1), 1, atol=1e-5)
    assert not np.allclose(embedding, embedding[0])
    assert trainer.get_model_info()['training_params']['epochs'] == 2
    np.testing.assert_array_equal(
        trainer.extract_latent_components(output)['combined_fusion'], embedding)
    assert output['mmvae_mu1'].shape == (36, 8)
