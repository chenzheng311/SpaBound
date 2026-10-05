# SpaBound examples

These notebooks demonstrate paired spatial multi-omics analysis from input H5AD files to graphs, embeddings, clustering, metrics and plots. Data and trained results are not bundled. Install SpaBound and the notebook dependencies using the repository README; mclust clustering additionally requires R, the R package `mclust`, and `rpy2`.

Start Jupyter with its working directory at the repository root or `examples/`. Open one notebook, select the installed Python environment, and run its cells from top to bottom in a fresh kernel.

## Input layout

Place your own files under the repository's `data/` directory, or set `SPABOUND_DATA_DIR` to a root with this structure:

```text
data/
  mouse_embryo/
    s1_adata_rna.h5ad
    s1_adata_atac.h5ad
  hln_d1/
    adata_RNA.h5ad
    adata_ADT.h5ad
  hln_a1/
    adata_RNA.h5ad
    adata_ADT.h5ad
```

Each pair must contain raw counts in `X`, unique matching spot identifiers in the same order, and finite spatial coordinates in `obsm['spatial']`. The notebooks reject mismatched spot order instead of silently reordering or dropping observations. Optional reference labels are `obs['Joint_clusters']` for the mouse embryo and `obs['ground_truth']` for HLN. Their absence skips reference-label metrics. The ATAC example reuses `obsm['X_lsi']` when supplied; otherwise it computes LSI.

All file locations and cluster counts are collected near the top of each notebook. Relative environment-variable paths are resolved from the repository root. `SPABOUND_OUTPUT_DIR` overrides the default `outputs/` root.

## Execution

1. Load and validate the paired inputs.
2. Preprocess RNA with gene filtering, 3,000 highly variable genes, count normalization, log transformation, scaling and PCA. Process ADT with CLR/scaling/PCA, or ATAC with LSI.
3. Construct boundary-aware spatial and feature graphs.
4. Train SpaBound, then cluster the row-normalized shared posterior means using PCA20 and mclust.
5. Evaluate available reference labels and visualize UMAP/spatial domains.
6. Save results to `outputs/<dataset>/`.

The output folder contains `spabound_result.h5ad`, `clusters.csv`, `metrics.csv`, and `configuration.json`. Input files are not overwritten. Repeating the save cell replaces the previous files in that example's output folder. Predictions are not aligned or renamed using reference labels; permutation-invariant metrics are calculated directly, with the evaluated spot count recorded.

## Example settings

The two HLN examples use the same graph and training settings, based on the D1 workflow. The A1 graph/training entry point has been completed for execution from a fresh kernel. The mouse embryo example preserves its source workflow's active configuration. These are runnable examples, not a claim of exact paper-table reproduction.

| Setting | Mouse embryo | HLN A1 and D1 |
|---|---:|---:|
| Learning rate | 0.0002 | 0.0005 |
| Weight decay | 0.0001 | 0.001 |
| Epochs | 300 | 800 |
| Encoder output width | 160 | 160 |
| Requested `z_dim` | 64 | 128 |
| Shared / private per modality | 32 / 16 | 32 / 12 |
| APPNP hidden width / steps / alpha | 128 / 2 / 0.25 | 160 / 2 / 0.15 |
| Spatial neighbors / beta | 6 / 0.8 | 24 / 0.8 |
| Spatial candidates / feature neighbors / beta | 40 / 15 / 0.6 | 20 / 8 / 1.2 |
| Reconstruction / KL / edge weights | 1 / 0.005 / 0.0001 | 1 / 0.02 / 0.0005 |
| Number of clusters | 14 | A1: 10; D1: 11 |

Both configurations use APPNP residual connections, zero dropout, automatic spatial bandwidth, multimodal multiplier 1, and the Charbonnier edge loss. The latent implementation reconciles `z_dim` with shared plus two private blocks: the effective total widths are 64 and 56, respectively. Auxiliary private coordinates are not presented as validated modality-specific biological factors.

Preprocessing uses seed 42, the trainer uses seed 2022, and the mclust helper uses R seed 2020. HLN PCA dimensionality is ADT feature count minus one, giving 30 PCs for 31 ADT markers. The embryo workflow retains its RNA PCA rule `min(50, ATAC_feature_count - 1)` and uses 51-component LSI with the first component removed when LSI is computed. Restart from the input-loading cell when changing data or settings; numerical reproducibility also depends on library versions and hardware.
