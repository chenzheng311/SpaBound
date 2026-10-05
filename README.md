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

## Validation

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The tests exercise graph construction, CPU training, and output integrity on
small synthetic data, as well as notebook structure. These remain installation
checks, separate from the complete example runs below.

All three examples were also run from counts on an NVIDIA GeForce RTX 4070 Ti
SUPER using the released model and fixed notebook settings. Each dataset had
one run, without parameter search or score-based reruns during verification.
The two HLN examples used identical graph and training settings.

| Dataset | Spots evaluated | Epochs | Observed ARI |
|---|---:|---:|---:|
| HLN A1 | 3,484 / 3,484 | 800 | 0.3707401885 |
| HLN D1 | 3,359 / 3,359 | 800 | 0.3582863835 |
| Mouse embryo | 2,186 / 2,186 | 300 | 0.4795129460 |

D1 agrees with the paper's ARI to four decimal places; A1 and mouse embryo are
higher in these runs. This verifies execution of the released examples, rather
than asserting exact reproduction of every historical paper workflow or
performance across seeds and hardware. See [execution results](docs/REPRODUCTION.md)
for NMI/AMI, seeds, software versions and protocol details,
[release notes](docs/RELEASE_NOTES.md) for packaging changes, and
[migration notes](docs/MIGRATION.md) for renamed modules and keys.

## License and attribution

This distribution retains the GNU AGPLv3 in [LICENSE](LICENSE), reflecting
SpatialGlue-derived components. The original shared-private multimodal VAE
repository's MIT notice is preserved separately in
[licenses/](licenses/shared-private-multimodalVAE-MIT.txt).
[NOTICE.md](NOTICE.md) records the implementation lineage and modifications.
