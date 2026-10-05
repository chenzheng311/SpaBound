# SpaBound naming and API migration

Use the `spabound` Python package after installing the repository. The original
workspace is not required at runtime.

| Previous module or interface | Published module or interface |
|---|---|
| `spatialVAE` | `spabound` |
| `SpatialGlue_MMVAE.py` | `spabound/trainer.py` |
| `Train_SpatialGlue_MMVAE` | `spabound.SpaBound` |
| `model_mmvae_appnp.py` | `spabound/model.py` |
| `SpatialMMVAE_APPNP` | `spabound.SpaBoundModel` |
| `SpatialMMVAE_APPNP_PrivateRecon` | `spabound.SpaBoundPrivateReconModel` |
| `preprocess.py` | `spabound/preprocessing.py` |
| `preprocess_boundary.py` | `spabound/graph.py` |
| `preprocess_boundary_v2.py` | `spabound/graph_v2.py` |
| `losses.py`, `utils.py` | `spabound/losses.py`, `spabound/utils.py` |
| `output['SpatialGlue_MMVAE']` | `output['SpaBound']` |
| Default clustering column `SpatialGlue` | `SpaBound` |

The renamed interfaces are used consistently in all published examples. Legacy
module/class aliases are not installed. Auxiliary output keys such as
`MMVAE_Shared`, `MMVAE_Private_1`, `MMVAE_Private_2`, and `mmvae_mu1` retain their
existing names to keep analyses of these tensors compatible.

## Minimal import migration

```python
from spabound import SpaBound, construct_boundary_aware_graphs_v2, clustering
from spabound.preprocessing import pca, lsi, clr_normalize_each_cell
```

The trainer still accepts the same parameter names. Graph settings are supplied
to the graph constructor, whose returned adjacency matrices are passed into the
trainer's data dictionary. The effective variational width is
`n_latent_shared + 2 * n_latent_private`; `z_dim` remains a compatibility argument.

## Notebook names

| Original notebook | Published notebook |
|---|---|
| `test_mouse_embryo_VAE_aware.ipynb` | `examples/spabound_mouse_embryo.ipynb` |
| `test_hln_d1_vae_aware.ipynb` | `examples/spabound_hln_d1.ipynb` |
| `test_hln_vae_aware.ipynb` | `examples/spabound_hln_a1.ipynb` |

Change file paths through the notebook configuration cells or the
`SPABOUND_DATA_DIR` and `SPABOUND_OUTPUT_DIR` environment variables. Original
data files are never used as output destinations.
