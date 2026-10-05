# SpaBound

SpaBound learns spatial domain representations from paired spatial multi-omics
data. It combines molecular-gradient-weighted spatial and feature graphs,
APPNP propagation, shared variational fusion, and edge-preserving regularization.
The examples cover RNA + ADT in human lymph nodes and RNA + ATAC in mouse embryo.

## Installation

Use Python 3.10 or 3.11; the release validation uses Python 3.11. Install the
appropriate CPU or CUDA build of PyTorch using the
[official installation instructions](https://pytorch.org/get-started/locally/).
Then, from this repository's root:

```bash
python -m pip install -e ".[examples]"
```

`requirements-tested.txt` records the direct Python dependency versions used
for local validation. To constrain an installation to those versions:

```bash
python -m pip install -c requirements-tested.txt -e ".[examples]"
```

The notebook examples use **mclust**, which additionally requires a working R
installation, the R package `mclust`, and `rpy2`:

```r
install.packages("mclust")
```

```bash
python -m pip install -e ".[mclust]"
```

Configure R for your environment before importing `rpy2`; no machine-specific
`R_HOME` is embedded in the package. See the
[rpy2 installation documentation](https://rpy2.github.io/doc.html).
R is needed only for mclust clustering, not for graph construction or training.
Leiden and Louvain can instead be installed with the `leiden` and `louvain`
extras; changing the clustering method changes the analysis protocol.

## Examples

| Notebook | Modalities |
|---|---|
| [HLN A1](examples/spabound_hln_a1.ipynb) | RNA + ADT |
| [HLN D1](examples/spabound_hln_d1.ipynb) | RNA + ADT |
| [Mouse embryo](examples/spabound_mouse_embryo.ipynb) | RNA + ATAC |

See [examples/README.md](examples/README.md) for the data layout. Place your
paired `.h5ad` files under `data/`, or set `SPABOUND_DATA_DIR`. Launch Jupyter
from the repository root and execute the chosen notebook from top to bottom:

```bash
jupyter lab
```

The HLN notebooks share a graph and training configuration. The mouse notebook
uses its own configuration. Each notebook exposes its seed, model settings,
graph settings, and target cluster count. Outputs are written under `outputs/`.
The notebooks contain no saved run outputs, and do not bundle benchmark data.

## Python interface

Once both modalities have been preprocessed, supply two aligned AnnData objects
with `.obsm["feat"]` and `.obsm["spatial"]`:

```python
import torch
from spabound import SpaBound, construct_boundary_aware_graphs_v2

graphs = construct_boundary_aware_graphs_v2(
    adata_rna,
    adata_other,
    n_neighbors_spatial=6,
    spatial_candidates_k=25,
    n_neighbors_feature=10,
    beta_spatial=1.0,
    beta_feature=1.0,
)
data = {"adata_omics1": adata_rna, "adata_omics2": adata_other, **graphs}
trainer = SpaBound(
    data,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    random_seed=2022,
    epochs=600,
    use_appnp=True,
    use_edge_preserving_loss=True,
)
result = trainer.train()
adata_rna.obsm["SpaBound"] = result["SpaBound"]
```

These settings illustrate the API; the notebooks specify their own complete
example configurations. Graphs are constructed explicitly before creating the
trainer. Its retained `use_boundary_aware_graph` option records configuration
metadata; it does not build or replace the supplied graphs.

`result["SpaBound"]` contains row-normalized shared posterior-mean features for
clustering. Additional output keys expose per-modality embeddings, attention
weights, and Gaussian parameters. The default reconstruction and spatial
regularization operate on the shared representation. Auxiliary private
coordinates are retained in the parameterization; their presence alone does
not establish identifiable disentanglement. The optional private-reconstruction
variant is not enabled in the examples.

The current trainer converts adjacency matrices to dense tensors. GPU memory
and RAM therefore grow quadratically with the number of spots. Inputs must
describe paired, co-registered spots; the examples check their alignment.

## Repository layout

```text
spabound/
  trainer.py         Training interface: SpaBound
  model.py           Neural modules: SpaBoundModel
  graph_v2.py        Continuous boundary-aware graph construction
  graph.py           Earlier graph variant, retained for compatibility
  preprocessing.py  PCA, CLR, LSI, and graph preprocessing
  losses.py         Losses and KL annealing
  utils.py          Clustering and plotting helpers
examples/           Three clean notebooks with explicit configurations
tests/              Small synthetic CPU smoke tests
docs/               Naming migration, release notes, and execution results
```


