# Attribution and distribution notices

SpaBound packages the supplied boundary-aware spatial integration implementation
under a consistent Python API. Packaging modifications are dated **2026-10-05**.

## SpatialGlue-derived components

The preprocessing, clustering, and several graph-network building blocks derive
from or adapt **SpatialGlue**, by Yahui Long and collaborators:

- [Official repository](https://github.com/JinmiaoChenLab/SpatialGlue)
- [Version 1.1.4 distribution](https://pypi.org/project/SpatialGlue/1.1.4/)
- [Upstream license file](https://github.com/JinmiaoChenLab/SpatialGlue/blob/main/LICENSE.md)

The official 1.1.4 source distribution contains the GNU Affero General Public
License, version 3. Its license text is preserved verbatim as this repository's
`docs/legal/AGPL-3.0.txt`. The distribution metadata's MIT label conflicts with that included
text; this package follows the included AGPLv3 text for the derived distribution.

The supplied preprocessing module closely follows SpatialGlue preprocessing;
clustering and plotting helpers also retain its implementation lineage. The
SpaBound model adds boundary-aware graphs, APPNP encoding and variational fusion;
the complete model is not presented as an unchanged copy of SpatialGlue.

## Shared-private multimodal VAE repository

The source workspace originated from the code accompanying **Disentangling
shared and private latent factors in multimodal Variational Autoencoders**.
Its MIT license and the copyright notice **Copyright (c) 2023 Kaspar Märtens**
are retained verbatim in `licenses/shared-private-multimodalVAE-MIT.txt`.
That notice applies to its respective upstream material; it does not replace
the SpatialGlue license above. Unused imports from the original sibling
`multimodalVAE` package were removed; it is not a runtime dependency.

## Modifications in this distribution

- Renamed the package, training/model classes, modules, and primary output key
  to SpaBound; made internal imports self-contained.
- Removed unused imports and local paths, and corrected metadata and helper API
  compatibility issues. The core training computations and defaults are retained.
- Reorganized three example notebooks, cleared saved outputs, and introduced
  configurable data/output paths. Both HLN examples use one shared configuration.
- Added installation metadata, documentation, and synthetic smoke tests.

See `docs/RELEASE_NOTES.md` for the specific compatibility changes. This package
retains the upstream warranty disclaimer in `docs/legal/AGPL-3.0.txt`.
