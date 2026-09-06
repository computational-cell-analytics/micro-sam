# Block-wise Tiled 3D APG

## Goal

Turn the current tiled 3D Automatic Prompt Generation (APG) implementation into a genuinely block-wise method that can distribute independent blocks across Z, Y, and X, and then recover global object identities from block overlaps with a multicut.

Use `bioimage_py` for the generic block orchestration and stitching. In particular, `bioimage_py.segmentation.stitch_segmentation` already implements haloed 3D tiling, temporary global instance IDs, overlap extraction between neighboring blocks, conversion of overlap evidence to multicut costs, multicut optimization, and projection into disjoint block cores. The APG-specific implementation should be limited to producing a local segmentation for one haloed block and managing the SAM GPU state efficiently.

The recommended design is:

```text
haloed XYZ blocks
    -> block-local APG instances
    -> bioimage_py overlap graph
    -> bioimage_py multicut
    -> bioimage_py relabeling of non-overlapping block cores
```

The central principle is that each inner block must produce a complete local segmentation. The halos intentionally produce redundant predictions, and the multicut turns the resulting block-local identities into global identities.

## Current limitation

At commit `07d1b05126bf855812399bc3120e7e2f6c324af2`, the core APG tiling is only in Y and X. Each APG tile is a full-depth column:

```python
volume[:, y0:y1, x0:x1]
```

The decoder uses overlapping Z blocks internally, but APG candidate ownership, SAM2 propagation, and final tile stitching do not. An individual propagation pass still traverses the complete Z extent.

The current generator also assigns each candidate to exactly one XY tile. This is incompatible with overlap-based identity stitching: if only one block predicts an object, adjacent blocks do not contain corresponding instance nodes for a multicut to join.

## 1. Use true 3D APG blocks

Introduce an APG block geometry with explicit Z, Y, and X components:

```text
block_shape = (block_z, block_y, block_x)
halo        = (halo_z, halo_y, halo_x)
```

Each block has:

- An **inner block**, which is the non-overlapping region owned by that block.
- An **outer block**, which is the inner block extended by its halo and clipped to the volume.

Each inference job operates on the outer block:

```python
volume[z0:z1, y0:y1, x0:x1]
```

but contributes only its inner block to the final segmentation.

### Embeddings

The existing embeddings can remain stored as XY tile columns because the SAM2 image encoder is applied slice-wise. A Z block can use a lazy view into the relevant slice range instead of encoding the Z halo again.

The block-local propagator needs a view that maps local frame indices to the corresponding global Z indices. It should expose only the outer block's Z range while preserving lazy reads from the existing Zarr-backed feature arrays.

### Propagator state

Generalize `TiledPromptableSegmentation3D` so that its state is keyed by a 3D `block_id`, not an XY `tile_id`. Its sub-volume and embeddings must both be restricted to the outer XYZ box. Candidate anchor frames are translated from global Z to block-local Z before prompting.

## 2. Run APG independently in overlapping blocks

Unique prompt ownership must be removed for block-wise APG. Neighboring blocks should deliberately predict the same object in their overlap.

For each outer block:

1. Crop the decoder prediction to the outer XYZ box.
2. Derive APG candidates within the crop.
3. Score the candidates on their local anchor slices.
4. Propagate through the outer block's Z range only.
5. Run the normal score-ordered local merge.
6. Keep local instances that intersect the block's inner core, while retaining their complete outer-block masks for overlap measurement.

The block result should contain at least:

```text
block_id
inner_box_zyx
outer_box_zyx
local instance segmentation over the outer block
APG score and stability per local instance
```

### Candidate coverage near block boundaries

A long object may have its global convergence point outside a block even though the object intersects the block's inner region. Purely routing the current global APG prompt would therefore leave some blocks without a local prediction.

A practical first implementation is to derive candidates independently from each haloed block. A robust fallback is to add a local interior candidate for any foreground component that intersects the inner block but has no regular density candidate. Candidate scoring can reject poor fallback prompts.

A more sophisticated alternative is to use the decoder flow to assign foreground voxels to convergence basins, then place one block-local interior prompt for each basin intersecting an inner block. This preserves the global candidate identities while still providing an independent seed in every relevant block.

## 3. Use the existing halo-aware stitching in `bioimage_py`

This proposal does **not** require a new halo-aware stitching algorithm in `micro-sam`. `bioimage_py` already exposes this functionality:

- `stitch_segmentation` runs a segmentation function independently on haloed blocks and compares the two predictions over the same physical voxels in their shared halo. This is the appropriate path for block-wise APG.

This is referred to below as "halo-aware stitching", already implemented by `bioimage_py.segmentation.stitch_segmentation`, **not to a separate replacement for `bioimage_py` stitching**.

The intended high-level integration is:

```python
import bioimage_py as bp


def segment_apg_block(block, block_id):
    # Return a dense instance segmentation for the complete haloed block.
    # Instance IDs only need to be unique within this block.
    return blockwise_apg(block, block_id)


segmentation = bp.segmentation.stitch_segmentation(
    input=volume,
    segmentation_function=segment_apg_block,
    tile_shape=block_shape,
    tile_overlap=halo,
    output=output,
    shape=volume.shape,
    with_background=True,
    beta=stitching_beta,
    num_workers=num_workers,
    job_type=job_type,
    job_config=job_config,
)
```

Here, `tile_shape` is the APG inner block shape and `tile_overlap` is the halo. Both are three-dimensional, so blocks can be scheduled independently across Z, Y, and X. The callback receives a complete haloed block and must return a label image of the same spatial shape; returning only the inner block would remove the evidence needed for stitching.

For each block, `bioimage_py` assigns globally unique temporary IDs to the block-local objects and writes the non-overlapping core. It compares the stored halo segmentations of face-adjacent blocks, builds a region adjacency graph over the assembled cores, and assigns overlap-derived costs to the corresponding region-adjacency edges. Edge- and corner-neighbor block pairs are unnecessary initially: agreement can propagate through face adjacencies, and direct diagonal matches tend to be less reliable.

For every pair of local instances with non-zero overlap in a shared halo, the standard `bioimage_py` stitching implementation computes directed overlap evidence from the intersection and label size in the overlap face. If that label pair also has an edge in the core region adjacency graph, it converts the strongest overlap observation into a disaffinity:

```text
disaffinity(u, v) = 1 - overlap_fraction(u, v)
```

Large overlap therefore produces a low disaffinity and strong merge evidence. The first APG implementation should use this existing behavior as its baseline. This separates the work needed for XYZ APG inference from possible improvements to the generic stitching algorithm.

## 4. Convert overlap evidence into multicut costs

The `bioimage_py` stitching code passes its overlap-derived disaffinities to its public cost transformation:

```python
costs = bp.segmentation.compute_edge_costs(
    disaffinities,
    beta=stitching_beta,
)
```

Positive costs favor joining nodes and negative costs favor cutting them. `stitching_beta` controls the global merge/cut prior while preserving the continuous strength of the overlap evidence. Start with one `beta` so that the implementation follows the standard `bioimage_py` path. Axis-specific priors or reliability factors should only be added to `bioimage_py` if measurements show that Z correspondences require different calibration from XY correspondences.

The graph is solved with `bioimage_py.segmentation.multicut_decomposition`. The complete operation is already part of `stitch_segmentation`; the explicit calls are useful only for testing or for a future precomputed-block API:

```python
node_labels = bp.segmentation.multicut_decomposition(
    graph,
    costs,
    n_threads=n_threads,
)
```

## 5. Improve block stitching generically in `bioimage_py`

The existing halo-aware implementation is the right starting point, but its graph and overlap model can be improved. These changes should be implemented in `bioimage_py` and exposed through `stitch_segmentation`, so that `micro-sam` continues to use the public stitching API and all other block-wise segmentation methods benefit from the same improvements.

The improvements are listed below in recommended priority order.

### 5.1 Build an explicit block-instance correspondence graph

The current stitcher first assembles the block cores and builds a region adjacency graph from this core segmentation. It then applies halo-overlap evidence only to label pairs that also form an edge in that region adjacency graph.

This can discard useful evidence. Two block-local instances may overlap strongly in the shared halo but fail to touch exactly at the core boundary because one prediction is eroded, shifted, or locally missing. They are then not adjacent in the assembled core segmentation, even though the halo provides a good identity match.

A better generic formulation is:

1. create one node for every block-local instance that contributes to a core;
2. add an edge for every supported correspondence measured in a shared halo, regardless of whether the two core masks touch exactly;
3. add only the required repulsive or compatibility edges between competing nodes;
4. solve this compact instance graph; and
5. project the component labels into the cores.

This uses the halo evidence directly and avoids constructing the stitching topology indirectly from voxel adjacency. It also makes graph size depend mainly on the number of block-local instances and overlap candidates rather than on a full-volume region adjacency computation.

### 5.2 Use symmetric and support-aware overlap confidence

The current directed fraction can give high confidence when a small fragment lies completely inside a much larger prediction. Compute both directed coverages,

```text
r_a = |A ∩ B| / |A|
r_b = |A ∩ B| / |B|
```

and combine them explicitly. Reasonable generic alternatives include:

- geometric mean, `sqrt(r_a * r_b) = |A ∩ B| / sqrt(|A| |B|)`, for balanced matching;
- Dice overlap, `2 |A ∩ B| / (|A| + |B|)`;
- `min(r_a, r_b)` or IoU for a stricter merge criterion.

Confidence should also depend on absolute support. An overlap of one voxel should not carry the same certainty as a large overlap with the same fraction. This can be handled through Bayesian smoothing, a minimum-support rule, or a reliability factor multiplying the edge log-odds. Axis-specific calibration may be useful for anisotropic data, but should be data-driven rather than hard-coded for APG.

### 5.3 Represent competing correspondences

A one-to-many overlap remains an important validation case:

```text
A1 -- B1
 |
 +--- B2
```

Purely attractive cross-block edges can merge `B1` and `B2` transitively even though they are distinct instances in the same block. Generic solutions include:

- soft repulsive, possibly lifted edges between same-block instances competing for the same neighbor;
- mutual-best or capacity-constrained correspondence filtering;
- a calibrated penalty for one-to-many assignments.

Soft repulsion is a good default because evidence from several blocks may legitimately correct a local over-segmentation. An absolute must-not-link would assume that every block-local segmentation is already correct.

### 5.4 Separate identity stitching from seam composition

The multicut decides which block-local instances have the same global identity; it does not decide which local boundary is spatially best. Copying disjoint cores is deterministic and often sufficient, but it can retain a visible seam if one core prediction is poor near the boundary.

An optional generic compositor could first map all halo predictions to global component IDs and then choose labels in the overlap by:

- distance-to-block-boundary weighted voting;
- prediction-confidence weighted voting; or
- a small seam optimization favoring boundaries in low-confidence regions.

This should be a separate `bioimage_py` option after identity resolution. Keeping it separate avoids mixing the graph's object-identity objective with voxel-level boundary selection.

### 5.5 Scale graph construction and optimization independently

The overlap-counting stages are already block-wise, whereas the current final region adjacency graph and multicut are coordinated globally. An explicit compact instance graph makes it possible to aggregate overlap edges block-wise, solve connected components independently where possible, and use decomposition for large connected subgraphs. These are general scalability improvements and also belong in `bioimage_py`.

None of these refinements is a prerequisite for the first APG version. The initial implementation should use the existing `stitch_segmentation` behavior and its tests as a baseline. Improvements should then be validated in `bioimage_py` on synthetic block-stitching cases before APG adopts them through a dependency update.

## 6. Solve globally and render block cores

After solving the multicut, `bioimage_py`:

1. Map every core-contributing `(block_id, local_instance_id)` node to its multicut component.
2. Relabel each block-local segmentation with the component labels.
3. Copy only the relabeled inner block into the global output.

The inner blocks partition the volume, so this rendering is deterministic and independent of worker completion or block iteration order. Halos are used as graph evidence, not painted into the output with a first-come-first-served rule. This relabeling and core projection is already implemented by `stitch_segmentation`.

If core-only projection leaves visible boundary artifacts, use the optional generic overlap compositor described above. It should be implemented in `bioimage_py`, while APG only supplies any APG-specific confidence values through a generic callback or metadata interface.

## 7. Parallel execution

The natural worker job is an entire XYZ block with all of its candidate passes. Keeping the block on one worker lets all passes reuse the block's embeddings, video-predictor state, and cached slice features.

`bioimage_py` supports local, subprocess, and Slurm execution for its block stages. Schedule blocks dynamically, with the estimated expensive blocks first. A useful cost estimate is:

```text
number of propagation passes * outer Z extent
```

Once Z is blocked, there should usually be enough independent jobs to keep all inference devices busy. Splitting one block across multiple workers should be a fallback for a dominant block or for cases with fewer blocks than workers, because it duplicates state construction and embedding reads.

The APG callback must not reconstruct the SAM model for every block. Each GPU worker should own a persistent predictor and reuse it across jobs. If the existing `micro-sam` GPU pool cannot be represented safely as a `stitch_segmentation` callback, use a two-phase integration:

1. run the haloed APG block jobs with the existing persistent workers and store their complete halo segmentations;
2. use a small public `bioimage_py` entry point for overlap extraction, multicut, relabeling, and core projection from these precomputed block results.

Such an entry point should be factored out of `stitch_segmentation` in `bioimage_py`; its private stitching code should not be copied into `micro-sam`. `stitch_tiled_segmentation` is not an equivalent substitute for this APG workflow because it compares interfaces in an already assembled non-overlapping label volume rather than comparing the two predictions over their shared halo.

## 8. Interaction with candidate pruning

Propagation-wave pruning should initially remain disabled while validating block stitching. Block-local pruning may remove a prediction that would otherwise provide useful overlap evidence to a neighboring block.

Once the basic method is stable, pruning can be applied independently inside each block before graph construction. Cross-block pruning should not be performed before the multicut because cross-block duplicates are intentional.

## 9. Suggested implementation structure

Keep the APG-specific layer narrow and delegate the generic work:

```text
micro-sam
    segment_apg_block(haloed_block, block_id) -> local labels
    persistent APG GPU worker management
    APG-specific configuration and validation

bioimage_py
    XYZ blocking and halo geometry
    local/subprocess/Slurm execution
    temporary global ID assignment
    overlap measurement between neighboring blocks
    overlap-to-cost conversion
    multicut optimization
    relabeling and core projection
```

The default path should be one call to `bioimage_py.segmentation.stitch_segmentation` with the APG block callback. The two-phase precomputed-block path changes only how block results are produced; graph construction, costs, solving, and projection remain `bioimage_py` responsibilities.

The main integration points in the current implementation are:

- `micro_sam/v2/prompt_based_segmentation.py`: replace full-Z tile-column states with outer XYZ block states.
- `micro_sam/v2/automatic_prompt_generation.py`: replace unique XY candidate ownership and full-Z propagation jobs with block-local APG jobs.
- `micro_sam/v2/propagation_pool.py`: pass block geometry and local Z ranges to workers.
- `micro_sam/v2/batched_inference.py`: reuse the existing slice-wise embeddings through lazy Z views.

## 10. Validation strategy

Start with synthetic cases that isolate stitching behavior:

- One object crossing only a Z seam.
- One object crossing only a Y or X seam.
- One object crossing multiple axes and several blocks.
- Two touching objects on a seam.
- One-to-many and many-to-one local segmentation disagreements.
- A strong halo correspondence whose core masks do not touch at the block boundary.
- A tiny intersection with a high directed overlap fraction but insufficient absolute support.
- A small accidental overlap that should remain cut.
- A local over-segmentation that evidence from neighboring blocks should merge.
- A single-block configuration that must reproduce the non-blocked result.
- Identical output for different worker counts and completion orders.

For real datasets, measure fragmentation and false-merge rates separately for Z and XY seams. Tune `stitching_beta` on held-out overlap pairs before evaluating the complete segmentation. Only introduce alternative overlap statistics, support weighting, or axis-specific calibration if the standard `bioimage_py` weighting shows a measurable failure mode.

The Z halo must be large enough for two independent block predictions to contain reliable shared object masks. The decoder's current `z_halo=2` is a decoder-context setting and should not automatically be reused as the APG stitching halo; the appropriate APG halo depends on object extent, Z spacing, and SAM2 propagation stability.
