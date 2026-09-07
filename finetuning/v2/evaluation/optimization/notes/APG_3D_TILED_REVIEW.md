# APG 3D Tiling Review

This review covers the changes on `apg-3d-tiling` relative to its merge base with `origin/dev`
(`2968c51ac153314b6280c477bd24aa4fec051260`). The branch adds blockwise XYZ automatic prompt
generation, per-block inference, halo-overlap stitching, multi-device execution, and shared
whole-volume normalization.

## Findings

### P1: Tiled APG is broken in the annotator

For tiled 2D images, including the current slice of a tiled volume, the annotator creates a
`TiledAutomaticPromptGenerator` and calls `set_state` with the former decoder/embedding state
(`micro_sam/sam_annotator/_widgets.py:4854`). The rewritten tiled generator instead requires a state
containing `image`, `tile_shape`, and `halo`, so this call raises immediately. Even if the state were
adapted, the widget subsequently calls `propose` and `select`, which the new tiled generator does not
implement.

Full-volume tiled 3D APG is explicitly rejected by a GUI guard, but tiled 2D APG is allowed through
and therefore hits this incompatible interface.

### P1: Halo matches are discarded when core masks do not touch

`TiledAutomaticPromptGenerator.generate` delegates stitching to
`bioimage_py.segmentation.stitch_segmentation` (`automatic_prompt_generation.py:3179`). In the
required `bioimage-py` 0.2.1 implementation, halo correspondences are only applied to pairs that are
also adjacent in the region adjacency graph built from the core-only label mosaic.

A synthetic reproduction with perfect halo correspondence but a one-pixel background gap at the
core seam left the two block labels separate. Slightly shifted independent block predictions can
therefore remain split even when the halo provides direct identity evidence. The stitching graph
should retain valid halo correspondences independently of core adjacency, either in the dependency
or in a branch-specific stitching implementation.

### P2: RGB volume/video preprocessing crashes

`_volume_normalization_bounds` computes percentiles on a sampled `(Z, Y, X, C)` array with
`keepdims=True` and no reduction axes (`batched_inference.py:41`). This returns bounds shaped
`(1, 1, 1, 1)`. Applying them to a `(Y, X, C)` frame introduces an extra leading dimension, after
which `_load_frame_as_tensor` fails while permuting three axes.

Whole-volume color bounds must remain per-channel while dropping the sampled Z dimension before
they are applied to individual frames.

### P2: Z-halo candidates are not protected from propagation-wave pruning

The tiled generator forwards only `self._halo[-2:]` as its protected margin
(`automatic_prompt_generation.py:3169`), and `_is_protected_from_pruning` examines only the anchor
mask's Y/X bounding box. With `propagation_waves > 1`, a candidate anchored in the Z halo but away
from a Y/X boundary can be pruned as a duplicate, even if its propagation is needed to establish an
identity across a Z-block seam.

The protection state and check need to include the candidate's anchor frame relative to the Z halo.

### P2: Cached embeddings do not distinguish custom normalization bounds

The new public `norm_bounds` argument changes the normalized input and therefore the stored encoder
features (`util.py:813`), but the embedding cache signature records only the generic preprocessing
policy. Calling `precompute_image_embeddings` again with the same image, model, tiling, and save path
but different parent-volume bounds silently reuses features computed with the previous bounds.

The actual bounds, or a stable digest of them, should be included in cache validation metadata.

### P2: The 2D APG factory silently drops tiled-generator options

`get_instance_segmentation_generator` forwards `**kwargs` to APG for `ndim == 3`, but not in its 2D
branch (`instance_segmentation.py:1517`). Options documented for the tiled generator, such as `beta`,
`workers_per_device`, and `execution`, are consequently ignored for tiled 2D APG. For example,
requesting `beta=.123` and `workers_per_device=3` still constructs a generator with defaults `0.5`
and `1`.

## Validation

- `git diff --check` passed.
- The focused non-GUI suite passed: 288 tests and 5 subtests, with 3 unrelated xFormers warnings.
- Annotator tests could not be collected because `napari` is not installed in the review environment.
- A direct `bioimage-py` 0.2.1 stitching reproduction confirmed the core-adjacency/halo-overlap issue.
- A direct RGB-frame preprocessing reproduction confirmed the dimensionality failure.

## Recommendation

Do not merge the branch until the tiled annotator regression and halo-correspondence loss are fixed.
The normalization, Z-halo pruning, cache-signature, and factory-forwarding issues should be addressed
in the same change because they affect supported inputs or newly exposed branch options.
