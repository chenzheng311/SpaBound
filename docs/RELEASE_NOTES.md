# SpaBound 0.1.0

Release preparation date: 2026-10-05.

## Scope

This release organizes the supplied model implementation and three example
workflows into a self-contained Python package. The core computational bodies,
objectives, graph formulas, and constructor defaults are retained. The original
experimental files are unchanged.

The two HLN examples share the active D1 graph/training configuration. The A1
source notebook had its graph and trainer constructors commented out; the
published version restores an executable entry point with the shared settings.
The mouse example preserves its active source configuration. Cluster counts
remain 10 for A1, 11 for D1, and 14 for mouse embryo.

## Packaging and compatibility changes

- Public package, model/trainer classes, primary embedding key, and example names
  use SpaBound. Refer to `MIGRATION.md` for the complete mapping.
- Removed unused imports of a sibling `multimodalVAE` package and associated
  `sys.path` manipulation.
- Removed an invalid `weight_factors` metadata lookup from `get_model_info()`.
- mclust uses an explicit R matrix and extracts the named `classification`
  result; it no longer activates the deprecated global NumPy/R converter.
- PCA helper imports sparse matrix types from SciPy's public namespace.
- The violin plot helper no longer passes unsupported `show=False` to Seaborn.
- Logs are in English. Graph-weight summaries describe numerical attenuation.
- Notebooks separate inputs and outputs, validate spot order, expose seeds and
  parameters, clear execution artifacts, and omit unrelated debug/replay cells.
- Reference-label metrics use permutation-invariant comparisons directly;
  predicted clusters are not relabeled using the reference annotations.

## Retained behavior

The trainer uses dense adjacency tensors. Its graph-related constructor options
do not replace externally supplied graphs. The default private coordinates have
KL regularization but are not used for private reconstruction. Optional
private-reconstruction and earlier graph variants remain available, but the
three examples use continuous V2 graphs and shared-only reconstruction.

The examples have completed execution checks on their biological datasets,
as documented below; they do not assert exact reproduction of every historical
paper workflow. No experimental data, model checkpoints, private paths, or
saved notebook outputs are included.

## Validation scope

Validation includes syntax checks, cleaned notebook structure, synthetic paired
graph construction, and short CPU training with APPNP and GCN encoder options.
The source distribution and wheel are checked for package completeness and
standalone importability. These small tests remain separate from full example
execution.

On 2026-10-05, all three released examples additionally completed their full
preprocessing, GPU training, mclust clustering, evaluation and plotting workflow.
All spots were evaluated. One fixed-seed run per dataset produced ARI
0.3707401885 (HLN A1), 0.3582863835 (HLN D1), and 0.4795129460 (mouse embryo).
A1 and D1 used the same graph/training configuration; the example parameters
and model implementation were unchanged during verification. There was no
parameter search or score-based rerun.

The runs used fresh count preprocessing, an RTX 4070 Ti SUPER, PyTorch 2.1.0
with CUDA 12.1, R 4.4.1 and mclust 6.1.2. D1 matches the reported paper ARI to
four decimal places, while the A1 and embryo results are higher. These are
single-run execution results, not estimates of multi-seed uncertainty or
cross-hardware reproducibility. See [REPRODUCTION.md](REPRODUCTION.md) for the
complete metric table, environment and verification scope.
