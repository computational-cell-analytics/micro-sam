"""Evaluate iterative interactive segmentation on multiple microscopy datasets.

Datasets:
- Mouse-Skull-Nuclei-CBG (3D nuclei, EmbedSeg)
- Lucchi (3D EM mitochondria)
- LIVECell (2D phase-contrast cells)
- CREMI (3D neuron segemntation)

Results - Lucchi (lucchi_test.h5, 36 objects):

Default model - native settings (center frame, bidirectional, no chunk):
  iter 0: mSA=0.2036  SA50=0.3585  SA75=0.2000
  iter 1: mSA=0.1930  SA50=0.3846  SA75=0.1803
  iter 2: mSA=0.1586  SA50=0.3585  SA75=0.1429
  iter 3: mSA=0.2062  SA50=0.4118  SA75=0.2000

Finetuned model (--first_frame --forward_only --chunk_size 8 --num_init_cond_frames 2):
  iter 0: mSA=0.4732  SA50=0.7143  SA75=0.5319
  iter 1: mSA=0.5162  SA50=0.8000  SA75=0.5319
  iter 2: mSA=0.5198  SA50=0.8000  SA75=0.5319
  iter 3: mSA=0.5217  SA50=0.8000  SA75=0.5319

Results - Generalist best.pt (multi-GPU, dataset_choice="both"), held-out test splits:

EmbedSeg Mouse-Skull-Nuclei-CBG (X2_right.tif, 45 objects):
  Generalist regimes: (a) first_frame+forward_only (--first_frame --forward_only --chunk_size 8
  --num_init_cond_frames 2); (b) center+bidirectional (drop --first_frame/--forward_only, keep chunk8/2cond).
  (b) matches the random-start-frame bidirectional training and stays monotonic, edging ahead by iter 3.
  Specialist (debug/skull/.../best.pt, skull only) uses regime (a). Default uses native settings
  (center frame, bidirectional, no chunk, 1 cond); it is untrained for iterative correction, so it degrades.
  All rows from the same pipeline (same min_size, volume).
  box (default):
    iter 0: mSA=0.4383 SA50=0.6667 SA75=0.4754
    iter 1: mSA=0.3143 SA50=0.5254 SA75=0.3433
    iter 2: mSA=0.3144 SA50=0.5254 SA75=0.3043
    iter 3: mSA=0.2857 SA50=0.5000 SA75=0.3043
  box (specialist):
    iter 0: mSA=0.4173 SA50=0.5789 SA75=0.5000
    iter 1: mSA=0.4388 SA50=0.6071 SA75=0.5517
    iter 2: mSA=0.4479 SA50=0.6071 SA75=0.5517
    iter 3: mSA=0.4421 SA50=0.6071 SA75=0.5254
  box (generalist, first_frame+forward_only):
    iter 0: mSA=0.4847 SA50=0.6667 SA75=0.5789
    iter 1: mSA=0.5113 SA50=0.6981 SA75=0.6071
    iter 2: mSA=0.5175 SA50=0.7308 SA75=0.6364
    iter 3: mSA=0.4971 SA50=0.6667 SA75=0.6071
  box (generalist, center+bidirectional):
    iter 0: mSA=0.4748 SA50=0.6667 SA75=0.5517
    iter 1: mSA=0.5112 SA50=0.6981 SA75=0.5789
    iter 2: mSA=0.5112 SA50=0.6981 SA75=0.5789
    iter 3: mSA=0.5153 SA50=0.6981 SA75=0.6071
  point (default):
    iter 0: mSA=0.3596 SA50=0.5789 SA75=0.3846
    iter 1: mSA=0.2957 SA50=0.5000 SA75=0.3043
    iter 2: mSA=0.2594 SA50=0.4754 SA75=0.2500
    iter 3: mSA=0.2526 SA50=0.4286 SA75=0.2329
  point (specialist):
    iter 0: mSA=0.3840 SA50=0.5517 SA75=0.4516
    iter 1: mSA=0.3978 SA50=0.5517 SA75=0.4754
    iter 2: mSA=0.4010 SA50=0.5517 SA75=0.4754
    iter 3: mSA=0.4412 SA50=0.6071 SA75=0.5254
  point (generalist, first_frame+forward_only):
    iter 0: mSA=0.4809 SA50=0.6667 SA75=0.5789
    iter 1: mSA=0.4821 SA50=0.6667 SA75=0.5789
    iter 2: mSA=0.4870 SA50=0.6667 SA75=0.6071
    iter 3: mSA=0.4754 SA50=0.6667 SA75=0.5789
  point (generalist, center+bidirectional):
    iter 0: mSA=0.4694 SA50=0.6667 SA75=0.5517
    iter 1: mSA=0.4542 SA50=0.6364 SA75=0.5254
    iter 2: mSA=0.4851 SA50=0.6667 SA75=0.5517
    iter 3: mSA=0.4864 SA50=0.6667 SA75=0.5517

LIVECell (20 images, SAM2ImagePredictor with logits between iters):
  Default and generalist share the 2D regime (no chunking in 2D); unlike skull, the default
  improves with corrections here because the image predictor's logit carry-over is in-distribution.
  box (default):
    iter 0: mSA=0.4159 SA50=0.8042 SA75=0.3971
    iter 1: mSA=0.4887 SA50=0.8626 SA75=0.4993
    iter 2: mSA=0.5483 SA50=0.9058 SA75=0.5855
    iter 3: mSA=0.5864 SA50=0.9278 SA75=0.6487
  box (generalist):
    iter 0: mSA=0.5687 SA50=0.9189 SA75=0.6321
    iter 1: mSA=0.6281 SA50=0.9549 SA75=0.7125
    iter 2: mSA=0.6700 SA50=0.9698 SA75=0.7721
    iter 3: mSA=0.6949 SA50=0.9747 SA75=0.8070
  point (default):
    iter 0: mSA=0.0630 SA50=0.1087 SA75=0.0701
    iter 1: mSA=0.1279 SA50=0.2265 SA75=0.1331
    iter 2: mSA=0.2101 SA50=0.3871 SA75=0.2112
    iter 3: mSA=0.3056 SA50=0.5757 SA75=0.2909
  point (generalist):
    iter 0: mSA=0.3178 SA50=0.5421 SA75=0.3489
    iter 1: mSA=0.4050 SA50=0.6633 SA75=0.4458
    iter 2: mSA=0.4800 SA50=0.7744 SA75=0.5212
    iter 3: mSA=0.5496 SA50=0.8542 SA75=0.6022

CREMI sample C ROI [0:16, 0:512, 0:512] (3D EM neurons, neuron_ids used directly, no connected
components -> 119 objects). VI split/merge, adapted Rand error (aRAND) and CREMI score: lower is
better (mSA/SA50/SA75 are uninformative for dense neurons and not reported). Default = native
regime; generalist = center+bidirectional, chunk8, 2cond. The default massively over-merges
(VI_merge ~4-5, aRAND ~0.84); the generalist separates the neurons (aRAND ~0.07-0.14).
  box (default):
    iter 0: VI_split=0.1598 VI_merge=4.2177 aRAND=0.8363 CREMI=1.9134
    iter 1: VI_split=0.1420 VI_merge=4.2088 aRAND=0.8365 CREMI=1.9077
    iter 2: VI_split=0.1359 VI_merge=4.1958 aRAND=0.8363 CREMI=1.9033
    iter 3: VI_split=0.1302 VI_merge=4.1903 aRAND=0.8362 CREMI=1.9008
  point (default):
    iter 0: VI_split=0.0573 VI_merge=4.6553 aRAND=0.8454 CREMI=1.9960
    iter 1: VI_split=0.0777 VI_merge=4.6087 aRAND=0.8453 CREMI=1.9903
    iter 2: VI_split=0.1753 VI_merge=4.6061 aRAND=0.8479 CREMI=2.0134
    iter 3: VI_split=0.1699 VI_merge=4.6139 aRAND=0.8482 CREMI=2.0143
  box (generalist):
    iter 0: VI_split=0.8516 VI_merge=0.7481 aRAND=0.1375 CREMI=0.4691
    iter 1: VI_split=0.7493 VI_merge=0.7127 aRAND=0.0779 CREMI=0.3375
    iter 2: VI_split=0.6706 VI_merge=0.6336 aRAND=0.0737 CREMI=0.3099
    iter 3: VI_split=0.6753 VI_merge=0.6480 aRAND=0.0727 CREMI=0.3101
  point (generalist):
    iter 0: VI_split=0.8970 VI_merge=0.7953 aRAND=0.0926 CREMI=0.3959
    iter 1: VI_split=0.6560 VI_merge=0.6165 aRAND=0.0694 CREMI=0.2972
    iter 2: VI_split=0.6587 VI_merge=0.6536 aRAND=0.0755 CREMI=0.3147
    iter 3: VI_split=0.6289 VI_merge=0.6587 aRAND=0.0886 CREMI=0.3378
"""

import os
import argparse
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import imageio.v3 as imageio

from skimage.measure import label as connected_components

import torch

from torch_em.util.segmentation import size_filter

from elf.evaluation import mean_segmentation_accuracy, cremi_score

from micro_sam.util import segmentation_to_one_hot
from micro_sam.v2.normalization import normalize_raw
from micro_sam.prompt_generators import IterativePromptGenerator
from micro_sam.v1.evaluation.inference import _get_batched_prompts, _get_batched_iterative_prompts

from micro_sam.v2.util import get_sam2_image_predictor, get_sam2_model, precompute_image_embeddings
from micro_sam.v2.evaluation.inference import _embedding_tensors_to_numpy


SKULL_DATA = "/mnt/vast-nhr/projects/cidas/cca/data/embedseg/Mouse-Skull-Nuclei-CBG/test"
LUCCHI_DATA = "/mnt/vast-nhr/projects/cidas/cca/data/lucchi"
LIVECELL_DATA = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"
CREMI_DATA = "/mnt/vast-nhr/projects/cidas/cca/data/cremi"

# Held-out CREMI ROI from sample C. Training used samples A and B for gradient updates and C only
# for validation, so a sub-volume of C is the cleanest held-out CREMI. A 16x512x512 crop keeps the
# object count tractable while matching the 512 xy training scale (2x upscale to 1024).
CREMI_TEST_ROI = np.s_[0:16, 0:512, 0:512]


def load_test_volumes_skull(min_size=10):
    """Load Mouse-Skull-Nuclei-CBG test volumes.

    Args:
        min_size: Minimum instance size in voxels.

    Returns:
        List of (filename, raw, labels) tuples with 3D arrays.
    """
    img_dir = os.path.join(SKULL_DATA, "images")
    mask_dir = os.path.join(SKULL_DATA, "masks")
    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".tif"):
            continue
        raw = imageio.imread(os.path.join(img_dir, fname))
        raw = normalize_raw(raw)
        mask_fname = fname.replace("X", "Y")
        labels = imageio.imread(os.path.join(mask_dir, mask_fname))
        labels = connected_components(labels).astype("uint32")
        labels = size_filter(labels, min_size=min_size)
        samples.append((fname, raw, labels))
        print(f"Loaded {fname}: raw {raw.shape}, {len(np.unique(labels)) - 1} objects")
    return samples


def load_test_volumes_lucchi(min_size=10):
    """Load Lucchi EM mitochondria test volume (binary labels -> instances via CC).

    Args:
        min_size: Minimum instance size in voxels.

    Returns:
        List of (filename, raw, labels) tuples with 3D arrays.
    """
    test_path = os.path.join(LUCCHI_DATA, "lucchi_test.h5")
    with h5py.File(test_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]
    raw = normalize_raw(raw)
    labels = connected_components(labels).astype("uint32")
    labels = size_filter(labels, min_size=min_size)
    fname = "lucchi_test.h5"
    print(f"Loaded {fname}: raw {raw.shape}, {len(np.unique(labels)) - 1} objects")
    return [(fname, raw, labels)]


def load_test_volumes_cremi(min_size=500):
    """Load a held-out CREMI sample C ROI (3D EM neurons).

    Samples A and B were used for training; C only for validation, so a sub-volume of C is the
    cleanest held-out CREMI test. The neuron_ids are used directly as instances (no connected
    components, which would wrongly split disconnected fragments of the same neuron); ids are only
    relabelled to consecutive integers so they fit in uint32.

    Args:
        min_size: Minimum instance size in voxels.

    Returns:
        List of (filename, raw, labels) tuples with 3D arrays.
    """
    test_path = os.path.join(CREMI_DATA, "sampleC.h5")
    with h5py.File(test_path, "r") as f:
        raw = f["volumes/raw"][CREMI_TEST_ROI]
        labels = f["volumes/labels/neuron_ids"][CREMI_TEST_ROI]
    raw = normalize_raw(raw)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(raw.shape).astype("uint32")
    labels = size_filter(labels, min_size=min_size)
    fname = "sampleC_roi.h5"
    print(f"Loaded {fname}: raw {raw.shape}, {len(np.unique(labels)) - 1} objects")
    return [(fname, raw, labels)]


def load_test_images_livecell(n_images=20, min_size=50):
    """Load a subset of LIVECell test images as 2D (H, W) arrays.

    Args:
        n_images: Number of test images to evaluate.
        min_size: Minimum instance size in pixels.

    Returns:
        List of (filename, raw, labels) tuples with shape (H, W).
    """
    from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths
    img_paths, seg_paths = get_livecell_paths(LIVECELL_DATA, split="test", download=False)
    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(len(img_paths), size=min(n_images, len(img_paths)), replace=False).tolist())
    samples = []
    for i in indices:
        fname = os.path.basename(img_paths[i])
        raw = normalize_raw(imageio.imread(img_paths[i]))
        labels = connected_components(imageio.imread(seg_paths[i])).astype("uint32")
        labels = size_filter(labels, min_size=min_size)
        samples.append((fname, raw, labels))
        print(f"Loaded {fname}: raw {raw.shape}, {len(np.unique(labels)) - 1} objects")
    return samples


def propagate_chunked(predictor, inference_state, z_start, num_frames, chunk_size, forward_only=False):
    """Propagate in fixed-length chunks so memory chain length matches training."""
    video_segments = {}

    chunk_start = z_start
    while chunk_start < num_frames:
        for out_frame, _, out_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=chunk_start, max_frame_num_to_track=chunk_size
        ):
            video_segments[out_frame] = (out_logits[0] > 0).cpu().numpy().squeeze()
        chunk_start += chunk_size

    if not forward_only:
        chunk_start = z_start
        while chunk_start >= 0:
            for out_frame, _, out_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=chunk_start, max_frame_num_to_track=chunk_size, reverse=True
            ):
                if out_frame not in video_segments:
                    video_segments[out_frame] = (out_logits[0] > 0).cpu().numpy().squeeze()
            chunk_start -= chunk_size

    return video_segments


def _extra_init_frames(obj_zs, z_prompt, num_init_cond_frames):
    """Pick evenly-spaced extra conditioning frames from obj_zs, excluding z_prompt.

    Returns a list of at most num_init_cond_frames-1 frame indices.
    """
    if num_init_cond_frames <= 1:
        return []
    candidates = [int(z) for z in obj_zs if int(z) != z_prompt]
    if not candidates:
        return []
    n_extra = min(num_init_cond_frames - 1, len(candidates))
    indices = np.round(np.linspace(0, len(candidates) - 1, n_extra)).astype(int)
    return [candidates[i] for i in indices]


def _jitter_box(box, H, W, rng, noise=0.1, noise_bound=20, image_size=1024):
    """Jitter box corners following SAM2's sample_box_points scheme.

    Each corner of the [x0, y0, x1, y1] box is perturbed by a uniform offset in
    +-min(noise * side_length, bound), where bound is noise_bound pixels at
    image_size resolution scaled to the frame. The result is clamped to the frame.
    Matches the box noising SAM2 applies to box prompts during training.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    bound = noise_bound * max(H, W) / image_size
    max_dx, max_dy = min(bw * noise, bound), min(bh * noise, bound)
    offsets = rng.uniform(-1, 1, size=4) * np.array([max_dx, max_dy, max_dx, max_dy])
    jittered = np.array([x0, y0, x1, y1]) + offsets
    jittered[[0, 2]] = np.clip(jittered[[0, 2]], 0, W - 1)
    jittered[[1, 3]] = np.clip(jittered[[1, 3]], 0, H - 1)
    return jittered.astype(box.dtype)


@torch.no_grad()
def segment_volume(
    raw, labels, predictor, n_iterations, use_box=False, chunk_size=None,
    first_frame=False, forward_only=False, num_init_cond_frames=1, box_jitter=False,
):
    """Run iterative 3D interactive segmentation for all objects in a volume.

    Args:
        raw: Float32 array of shape (Z, H, W).
        labels: Integer label array of shape (Z, H, W).
        predictor: SAM2 video predictor.
        n_iterations: Number of prompt/correction rounds.
        use_box: Use bounding-box prompts instead of points.
        chunk_size: Propagation chunk size (None = full video).
        first_frame: Prompt from the first z-frame of each object (vs center).
        forward_only: Propagate forward only from the prompt frame.
        num_init_cond_frames: Number of frames to condition on before first propagation.
        box_jitter: Jitter the box prompt like SAM2 training (only used with use_box).

    Returns:
        List of segmentation arrays (one per iteration), each of shape (Z, H, W).
    """
    volume_embeddings = _embedding_tensors_to_numpy(
        precompute_image_embeddings(predictor=predictor, input_=raw, ndim=3)
    )
    inference_state = predictor.init_state(volume=raw, volume_embeddings=volume_embeddings)

    prompt_generator = IterativePromptGenerator()
    gt_ids = sorted(np.unique(labels)[1:].tolist())
    rng = np.random.default_rng(0)

    seg_per_iter = [np.zeros_like(labels) for _ in range(n_iterations)]

    for obj_id in tqdm(gt_ids, desc="Objects", leave=False):
        gt_3d = labels == obj_id
        obj_zs = np.where(gt_3d.any(axis=(1, 2)))[0]

        if first_frame:
            z_prompt = int(obj_zs.min())
        else:
            # Pick the middle slice among those the object actually appears in. For objects that are
            # disconnected across z (e.g. CREMI neurons without connected components), the geometric
            # midpoint can land in a gap with no object, giving an empty prompt slice.
            z_prompt = int(obj_zs[len(obj_zs) // 2])

        gt_slice = (labels[z_prompt] == obj_id).astype("uint32")
        points, point_labels, boxes = _get_batched_prompts(
            gt=gt_slice, gt_ids=[1],
            use_points=not use_box, use_boxes=use_box,
            n_positives=0 if use_box else 1, n_negatives=0, dilation=5,
        )
        if use_box and box_jitter:
            boxes = boxes.copy()
            boxes[0] = _jitter_box(boxes[0], labels.shape[1], labels.shape[2], rng)

        corr_points = corr_labels = None
        corr_frame = None

        for iteration in range(n_iterations):
            if iteration == 0:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z_prompt, obj_id=obj_id,
                    points=points, labels=point_labels,
                    box=boxes[0] if use_box else None,
                )
                for z_extra in _extra_init_frames(obj_zs, z_prompt, num_init_cond_frames):
                    gt_extra = (labels[z_extra] == obj_id).astype("uint32")
                    extra_pts, extra_lbls, _ = _get_batched_prompts(
                        gt=gt_extra, gt_ids=[1],
                        use_points=True, use_boxes=False,
                        n_positives=1, n_negatives=0, dilation=5,
                    )
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=z_extra, obj_id=obj_id,
                        points=extra_pts, labels=extra_lbls, box=None,
                    )
            else:
                for pt, lbl in zip(corr_points, corr_labels):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=corr_frame, obj_id=obj_id,
                        points=np.array([pt]), labels=np.array([lbl]),
                        box=None, clear_old_points=False,
                    )

            if chunk_size is not None:
                video_segments = propagate_chunked(
                    predictor, inference_state, z_prompt, labels.shape[0], chunk_size,
                    forward_only=forward_only,
                )
            else:
                video_segments = {}
                for out_frame, _, out_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=z_prompt
                ):
                    video_segments[out_frame] = (out_logits[0] > 0).cpu().numpy().squeeze()

                if not forward_only and len(video_segments) < labels.shape[0]:
                    for out_frame, _, out_logits in predictor.propagate_in_video(
                        inference_state, start_frame_idx=z_prompt, reverse=True
                    ):
                        if out_frame not in video_segments:
                            video_segments[out_frame] = (out_logits[0] > 0).cpu().numpy().squeeze()

            seg_3d = np.zeros(labels.shape, dtype=bool)
            height, width = labels.shape[1:]
            for z, mask in video_segments.items():
                seg_3d[z] = mask[:height, :width]

            seg_per_iter[iteration][seg_3d] = obj_id

            if iteration < n_iterations - 1:
                errors = np.array([np.sum(gt_3d[z] != seg_3d[z]) for z in obj_zs])
                z_worst = int(obj_zs[np.argmax(errors)])

                next_coords, next_labels = _get_batched_iterative_prompts(
                    sampled_binary_gt=torch.from_numpy(gt_3d[z_worst].astype("int64"))[None, None].float(),
                    masks=torch.from_numpy(seg_3d[z_worst].astype("int64"))[None, None].float(),
                    batch_size=32,
                    prompt_generator=prompt_generator,
                )
                corr_points = next_coords[0].cpu().numpy()
                corr_labels = next_labels[0].cpu().numpy()
                corr_frame = z_worst

        predictor.reset_state(inference_state)

    return seg_per_iter


@torch.no_grad()
def segment_image_2d(
    raw, labels, predictor, n_iterations, use_box=False, batch_size=32, box_jitter=False,
):
    """Run iterative 2D interactive segmentation using SAM2ImagePredictor.

    Closely follows the existing micro_sam.v2.evaluation.inference implementation:
    all objects in an image are batched together, correction points accumulate
    across iterations, and logits are optionally passed as mask_input.

    Args:
        raw: Float32 array of shape (H, W), normalized [0, 1].
        labels: Integer label array of shape (H, W).
        predictor: SAM2ImagePredictor.
        n_iterations: Number of prompt/correction rounds.
        use_box: Use bounding-box prompts instead of points.
        batch_size: Number of objects per inference batch.
        box_jitter: Jitter the box prompts like SAM2 training (only used with use_box).
    Returns:
        List of segmentation arrays (one per iteration), each of shape (H, W).
    """
    img_uint8 = (raw * 255).astype("uint8")
    if img_uint8.ndim == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)
    predictor.set_image(img_uint8)

    prompt_generator = IterativePromptGenerator()
    gt_ids = np.unique(labels)[1:]

    use_points = not use_box
    use_boxes = use_box
    n_positive = 0 if use_box else 1
    multimasking = not use_box

    points, point_labels, boxes = _get_batched_prompts(
        gt=labels, gt_ids=gt_ids,
        use_points=use_points, use_boxes=use_boxes,
        n_positives=n_positive, n_negatives=0, dilation=5,
    )
    if use_boxes and box_jitter:
        rng = np.random.default_rng(0)
        H, W = labels.shape
        boxes = boxes.copy()
        for i in range(len(boxes)):
            boxes[i] = _jitter_box(boxes[i], H, W, rng)

    sampled_binary_y = segmentation_to_one_hot(
        segmentation=labels.astype("int64"), segmentation_ids=gt_ids
    )

    logits_masks = None
    seg_per_iter = []

    for iteration in range(n_iterations):
        n_prompts = boxes.shape[0] if use_boxes else points.shape[0]
        n_batches = int(np.ceil(n_prompts / batch_size))

        all_masks, all_logits = [], []
        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, n_prompts)
            batch_points = points[s:e] if use_points or points is not None else None
            batch_point_labels = point_labels[s:e] if use_points or point_labels is not None else None
            batch_boxes = boxes[s:e] if use_boxes else None
            batch_logits = logits_masks[s:e] if logits_masks is not None else None

            batch_masks, batch_scores, batch_logits_out = predictor.predict(
                point_coords=batch_points,
                point_labels=batch_point_labels,
                box=batch_boxes,
                mask_input=batch_logits,
                multimask_output=multimasking,
            )

            if batch_scores.ndim == 2:
                max_idx = batch_scores.argmax(axis=1)
            else:
                max_idx = np.array([batch_scores.argmax()])
                batch_masks = batch_masks[None]
                batch_logits_out = batch_logits_out[None]

            if multimasking:
                batch_masks = np.stack([batch_masks[i, mid][None] for i, mid in enumerate(max_idx)])
                batch_logits_out = np.stack([batch_logits_out[i, mid][None] for i, mid in enumerate(max_idx)])

            all_masks.append(batch_masks)
            all_logits.append(batch_logits_out)

        masks = np.concatenate(all_masks)  # (N_obj, 1, H, W)
        logits_masks = np.concatenate(all_logits)
        multimasking = False

        seg = np.zeros(labels.shape, dtype="uint32")
        for obj_idx, obj_id in enumerate(gt_ids):
            seg[masks[obj_idx, 0] > 0] = obj_id
        seg_per_iter.append(seg)

        if iteration < n_iterations - 1:
            next_coords, next_labels = _get_batched_iterative_prompts(
                sampled_binary_gt=sampled_binary_y,
                masks=torch.from_numpy(masks).to(torch.float32),
                batch_size=batch_size,
                prompt_generator=prompt_generator,
            )
            next_coords = next_coords.detach().cpu().numpy()
            next_labels = next_labels.detach().cpu().numpy()
            points = np.concatenate([points, next_coords], axis=1) if points is not None else next_coords
            if point_labels is not None:
                point_labels = np.concatenate([point_labels, next_labels], axis=1)
            else:
                point_labels = next_labels

    return seg_per_iter


def evaluate_volume(labels, seg_per_iter, extra_metrics=False):
    """Compute per-iteration segmentation accuracy metrics.

    Args:
        labels: Ground-truth label array.
        seg_per_iter: List of segmentation arrays, one per iteration.
        extra_metrics: Report the CREMI metrics (VI split/merge, adapted Rand error and CREMI
            score) instead of mSA/SA50/SA75. Used for dense EM neuron segmentation, where the
            instance-matching scores are not informative.

    Returns:
        List of dicts with 'iteration' and either 'mSA'/'SA50'/'SA75' (default) or
        'VI_split'/'VI_merge'/'aRAND'/'CREMI' (when extra_metrics is set).
    """
    rows = []
    for i, seg in enumerate(seg_per_iter):
        if extra_metrics:
            vi_split, vi_merge, arand, cremi = cremi_score(seg, labels)
            row = {
                "iteration": i,
                "VI_split": round(vi_split, 4),
                "VI_merge": round(vi_merge, 4),
                "aRAND": round(arand, 4),
                "CREMI": round(cremi, 4),
            }
        else:
            msa, sa = mean_segmentation_accuracy(seg, labels, return_accuracies=True)
            row = {"iteration": i, "mSA": round(msa, 4), "SA50": round(sa[0], 4), "SA75": round(sa[5], 4)}
        rows.append(row)
    return rows


def get_predictor(model_type, backbone, checkpoint_path):
    """Build the SAM2 video predictor, handling both stock and micro-sam checkpoints.

    torch_em's DefaultTrainer saves {"model_state": ..., "epoch": ...}, while SAM2's
    _load_checkpoint expects {"model": ...} with weights_only=True. For micro-sam
    checkpoints we build the base model first and then load the weights manually.
    """
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state" in ckpt:
            predictor = get_sam2_model(model_type=model_type, backbone=backbone, input_type="videos")
            predictor.load_state_dict(ckpt["model_state"])
            return predictor
    return get_sam2_model(
        model_type=model_type, backbone=backbone, checkpoint_path=checkpoint_path, input_type="videos"
    )


def get_image_predictor(model_type, backbone, checkpoint_path):
    """Build SAM2ImagePredictor, handling both stock and micro-sam checkpoints.

    For micro-sam checkpoints (model_state key), loads weights via the video predictor
    (same underlying architecture) then wraps in SAM2ImagePredictor.
    """
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state" in ckpt:
            video_pred = get_sam2_model(model_type=model_type, backbone=backbone, input_type="videos")
            video_pred.load_state_dict(ckpt["model_state"])
            return get_sam2_image_predictor(video_pred)
    model = get_sam2_model(model_type=model_type, backbone=backbone, checkpoint_path=checkpoint_path)
    return get_sam2_image_predictor(model)


def run_eval_3d(dataset, samples, predictor, args):
    """Run 3D video-predictor evaluation and print results.

    Args:
        dataset: Dataset name string, used in output labels.
        samples: List of (filename, raw, labels) 3D tuples.
        predictor: SAM2 video predictor.
        args: Parsed argparse namespace.
    """
    all_rows = []
    for fname, raw, labels in samples:
        print(f"\nVolume: {fname}")
        seg_per_iter = segment_volume(
            raw, labels, predictor, n_iterations=args.n_iterations,
            use_box=args.prompt == "box", chunk_size=args.chunk_size,
            first_frame=args.first_frame, forward_only=args.forward_only,
            num_init_cond_frames=args.num_init_cond_frames, box_jitter=args.box_jitter,
        )
        rows = evaluate_volume(labels, seg_per_iter, extra_metrics=args.extra_metrics)
        for row in rows:
            row["volume"] = fname
        all_rows.extend(rows)
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    if len(samples) > 1:
        print(f"\nMean across {dataset} volumes:")
        df_all = pd.DataFrame(all_rows)
        print(df_all.groupby("iteration")[["mSA", "SA50", "SA75"]].mean().round(4).to_string())


def run_eval_2d(dataset, samples, predictor, args):
    """Run 2D image-predictor evaluation and print results.

    Args:
        dataset: Dataset name string, used in output labels.
        samples: List of (filename, raw, labels) 2D tuples.
        predictor: SAM2ImagePredictor.
        args: Parsed argparse namespace.
    """
    all_rows = []
    for fname, raw, labels in tqdm(samples, desc="Images"):
        print(f"\nImage: {fname}")
        seg_per_iter = segment_image_2d(
            raw, labels, predictor, n_iterations=args.n_iterations,
            use_box=args.prompt == "box", box_jitter=args.box_jitter,
        )
        rows = evaluate_volume(labels, seg_per_iter, extra_metrics=args.extra_metrics)
        for row in rows:
            row["volume"] = fname
        all_rows.extend(rows)
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    if len(samples) > 1:
        print(f"\nMean across {dataset} images:")
        df_all = pd.DataFrame(all_rows)
        print(df_all.groupby("iteration")[["mSA", "SA50", "SA75"]].mean().round(4).to_string())


def main():
    # NOTE: THINGS TAKEN CARE OF
    # 1. Normalization works fine, as expected.
    #   - i.e. [0, 1] -> normalize with ImageNet stats
    # 2. Random frame prompting + bidirectional propagation at training works as expected.
    #   - i.e. exactly how a user would like to do prompting.
    # 3. Resizing image, final design: resizelongestside(1024) -> pad rest to 1024 in both axes.
    # 4. Propagate over chunks (fairly critical).
    # 6. They propagate the masks further externally from previous predictions! (align train and eval)

    # NOTE: DOES NOT WORK AS EXPECTED
    # 5. Iterative rectification design.
    #   a. Current: put mask / box / point in the "first frame" -> randomly rectifies n frames in one go with n points -> that's it.  # noqa
    #   b. Expectation: iterate over n times, which is controllable.

    # TODO:
    # a) Work with boxes over iterations for new corrections as well.

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", default="hvit_t")
    parser.add_argument("-b", "--backbone", default="sam2.1")
    parser.add_argument("-c", "--checkpoint", default=None)
    parser.add_argument("-n", "--n_iterations", type=int, default=4)
    parser.add_argument("--min_size", type=int, default=10)
    parser.add_argument("--prompt", choices=["point", "box"], default="point")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--first_frame", action="store_true", help="Prompt from first z-frame of each object.")
    parser.add_argument("--forward_only", action="store_true", help="Propagate forward only from the prompt frame.")
    parser.add_argument("--box_jitter", action="store_true", help="Jitter box prompts like SAM2 training.")
    parser.add_argument(
        "--num_init_cond_frames", type=int, default=1,
        help="Frames to condition on before first propagation (1 = prompt only; 2 matches training).",
    )
    parser.add_argument(
        "--dataset", choices=["skull", "lucchi", "livecell", "cremi"], default=None,
        help="Dataset to evaluate (default: all three sequentially).",
    )
    parser.add_argument("--n_eval_images", type=int, default=20, help="Number of LIVECell test images to evaluate.")
    parser.add_argument("--extra_metrics", action="store_true", help="CREMI metrics: VI split/merge, aRAND, score.")
    args = parser.parse_args()

    run_datasets = [args.dataset] if args.dataset else ["skull", "lucchi", "livecell"]
    for ds in run_datasets:
        print(f"\n=== Evaluating {ds} ===")
        if ds == "skull":
            samples = load_test_volumes_skull(min_size=args.min_size)
            predictor = get_predictor(args.model_type, args.backbone, args.checkpoint)
            predictor.add_all_frames_to_correct_as_cond = True
            run_eval_3d(ds, samples, predictor, args)
        elif ds == "lucchi":
            samples = load_test_volumes_lucchi(min_size=args.min_size)
            predictor = get_predictor(args.model_type, args.backbone, args.checkpoint)
            predictor.add_all_frames_to_correct_as_cond = True
            run_eval_3d(ds, samples, predictor, args)
        elif ds == "livecell":
            samples = load_test_images_livecell(n_images=args.n_eval_images, min_size=args.min_size)
            predictor = get_image_predictor(args.model_type, args.backbone, args.checkpoint)
            run_eval_2d(ds, samples, predictor, args)
        elif ds == "cremi":
            samples = load_test_volumes_cremi(min_size=args.min_size)
            predictor = get_predictor(args.model_type, args.backbone, args.checkpoint)
            predictor.add_all_frames_to_correct_as_cond = True
            run_eval_3d(ds, samples, predictor, args)


if __name__ == "__main__":
    main()
