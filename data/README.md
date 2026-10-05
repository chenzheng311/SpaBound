# Input data

The repository distributes code and examples, without experimental data.

Each example loads two `.h5ad` files representing paired modalities. See
`examples/README.md` for the filenames and dataset subdirectories. You can place
the files here or set `SPABOUND_DATA_DIR` to another data directory.

Both AnnData objects must contain the same spots, in the same order, and their
spatial coordinates in `.obsm["spatial"]`. The examples prepare model features
in `.obsm["feat"]`. RNA and ADT inputs should contain counts before the example
normalization steps; the mouse example also prepares ATAC features by LSI.

Reference annotations are optional for model fitting. Evaluation cells require
the annotation column documented in the corresponding notebook.

Data files are excluded by `.gitignore`.
