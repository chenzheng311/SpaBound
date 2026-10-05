# Example execution results

Validation date: 2026-10-05. Each of the three released notebooks completed
one full run from counts using the released model and its fixed example
configuration. No parameter search or score-based reruns were performed during
this verification.

## Observed results

| Dataset | Spots evaluated | Epochs | ARI | NMI | AMI | Paper ARI |
|---|---:|---:|---:|---:|---:|---:|
| HLN A1 | 3,484 / 3,484 | 800 | 0.3707401885 | 0.4454659944 | 0.4413430385 | 0.3614 |
| HLN D1 | 3,359 / 3,359 | 800 | 0.3582863835 | 0.3829390100 | 0.3774741816 | 0.3583 |
| Mouse embryo | 2,186 / 2,186 | 300 | 0.4795129460 | 0.6155877536 | 0.6088221992 | 0.4651 |

D1 matches the paper ARI to four decimal places. A1 and mouse embryo have
higher ARI in these runs. These results validate the released example
configurations; they do not imply that every historical paper run used the
same configuration or execution environment. In particular, the released HLN
examples share the D1 configuration, including the completed A1 entry point
described in the [release notes](RELEASE_NOTES.md).

## Run protocol

- One fresh process per dataset, with preprocessing seed 42, trainer seed 2022,
  and mclust R seed 2020. The table contains individual runs, not seed averages.
- All input `X` matrices were checked to be finite, nonnegative and
  integer-valued. Fresh in-memory AnnData objects retained counts, spot/gene
  metadata and spatial coordinates; cached embeddings and preprocessing
  representations were discarded. RNA PCA and ATAC LSI were recomputed.
- A1 and D1 used identical graph and training dictionaries with 800 epochs;
  mouse embryo used its own released configuration with 300 epochs. See
  [example settings](../examples/README.md#example-settings) for the parameters.
  Target cluster counts were 10, 11 and 14, respectively.
- The shared posterior-mean representation was row-normalized, reduced with
  PCA20 and clustered using mclust. All paired spots were evaluated; predicted
  labels were not aligned to the reference labels before calculating metrics.
- Runtime adaptations were limited to input/output paths, the fresh count
  objects, noninteractive figure capture and output H5AD compression. Model,
  graph and training settings remained unchanged.

All 10 code cells completed in each notebook. Source hashes confirmed that
the executed model files matched the frozen release code. Training logs reached
800/800, 800/800 and 300/300 epochs on CUDA. Metrics were independently
recalculated from saved cluster labels and agreed with the recorded results
within 1e-12; saved H5AD labels and embeddings were also cross-checked. Input
file hashes were identical before and after execution.

## Recorded environment

| Component | Version or device |
|---|---|
| OS | Windows 10, build 26100 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| Python | 3.11.3 |
| PyTorch / CUDA runtime | 2.1.0 / 12.1 |
| NumPy / SciPy | 1.26.4 / 1.14.1 |
| pandas / scikit-learn | 2.2.3 / 1.7.0 |
| Scanpy / AnnData | 1.9.3 / 0.9.1 |
| scikit-misc / umap-learn | 0.5.2 / 0.5.3 |
| Matplotlib | 3.9.2 |
| R / mclust | 4.4.1 / 6.1.2 |
| rpy2 | 3.4.1 |

The direct Python dependency versions are also listed in
[`requirements-tested.txt`](../requirements-tested.txt). See the repository
[README](../README.md#installation) for installation and R setup.

## Scope and running the examples

These checks establish that the released workflows execute on the stated
inputs and environment. One run per dataset does not establish seed stability,
statistical significance or identical results on other hardware/library
versions. The synthetic CPU tests remain useful installation checks and are
separate from these complete biological-data runs.

Provide your own paired input files following the
[data layout](../examples/README.md#input-layout), then run a notebook from top
to bottom. The public notebooks are cleared of execution outputs. Experimental
data, trained weights and executed notebooks are not distributed with this
repository.
