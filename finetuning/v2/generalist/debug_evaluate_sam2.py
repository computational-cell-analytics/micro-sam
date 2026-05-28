"""Evaluate iterative interactive segmentation on multiple microscopy datasets.

Datasets:
- Mouse-Skull-Nuclei-CBG (3D nuclei, EmbedSeg)
- Lucchi (3D EM mitochondria)
- LIVECell (2D phase-contrast cells)

Results - Mouse-Skull-Nuclei-CBG (X2_right.tif, 45 objects):

Default model - native settings (center frame, bidirectional, no chunk):
  iter 0: mSA=0.3596  SA50=0.5789  SA75=0.3846
  iter 1: mSA=0.3062  SA50=0.5254  SA75=0.3235
  iter 2: mSA=0.2716  SA50=0.4516  SA75=0.2857
  iter 3: mSA=0.2580  SA50=0.4754  SA75=0.2329
  (corrections degrade - not trained for iterative regime)

Finetuned model (--first_frame --forward_only --chunk_size 8 --num_init_cond_frames 2):
  iter 0: mSA=0.4133  SA50=0.6071  SA75=0.4754
  iter 1: mSA=0.4385  SA50=0.6364  SA75=0.5000
  iter 2: mSA=0.4385  SA50=0.6364  SA75=0.5000
  iter 3: mSA=0.4600  SA50=0.6667  SA75=0.5254
  (consistent improvement with each correction round)

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

Results - LIVECell (20 images, mean, SAM2ImagePredictor with logits between iters):

Default model:
  iter 0: mSA=0.0909  SA50=0.1634  SA75=0.0942
  iter 1: mSA=0.1536  SA50=0.2641  SA75=0.1663
  iter 2: mSA=0.2339  SA50=0.4094  SA75=0.2480
  iter 3: mSA=0.3295  SA50=0.6000  SA75=0.3261

Finetuned model:
  iter 0: mSA=0.3905  SA50=0.6607  SA75=0.4168
  iter 1: mSA=0.4805  SA50=0.7776  SA75=0.5208
  iter 2: mSA=0.5533  SA50=0.8506  SA75=0.6115
  iter 3: mSA=0.6118  SA50=0.9041  SA75=0.6829
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

from sam2.sam2_image_predictor import SAM2ImagePredictor

from torch_em.transform.raw import normalize
from torch_em.util.segmentation import size_filter

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.util import segmentation_to_one_hot
from micro_sam.prompt_generators import IterativePromptGenerator
from micro_sam.evaluation.inference import _get_batched_prompts, _get_batched_iterative_prompts

from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings
from micro_sam.v2.evaluation.inference import _embedding_tensors_to_numpy


SKULL_DATA = "/mnt/vast-nhr/projects/cidas/cca/data/embedseg/Mouse-Skull-Nuclei-CBG/test"
LUCCHI_DATA = "/mnt/vast-nhr/projects/cidas/cca/data/lucchi"
LIVECELL_DATA = "/mnt/vast-nhr/projects/cidas/cca/data/livecell"


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
        raw = normalize(raw).astype(np.float32)
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
    raw = normalize(raw.astype(np.float32))
    labels = connected_components(labels).astype("uint32")
    labels = size_filter(labels, min_size=min_size)
    fname = "lucchi_test.h5"
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
        raw = normalize(imageio.imread(img_paths[i]).astype(np.float32))
        labels = connected_components(imageio.imread(seg_paths[i])).astype("uint32")
        labels = size_filter(labels, min_size=min_size)
        samples.append((fname, raw, labels))
        print(f"Loaded {fname}: raw {raw.shape}, {len(np.unique(labels)) - 1} objects")
    return samples


def pad_to_square(vol):
    """Pad (Z, H, W) volume on the right/bottom to make frames square.

    Matches torch_em's ensure_patch_shape convention: content starts at (0, 0),
    zeros are appended on the right (if W < H) or bottom (if H < W).
    """
    _, H, W = vol.shape
    if H == W:
        return vol
    if H > W:
        return np.pad(vol, ((0, 0), (0, 0), (0, H - W)))
    return np.pad(vol, ((0, 0), (0, W - H), (0, 0)))


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


@torch.no_grad()
def segment_volume(
    raw, labels, predictor, n_iterations, use_box=False, chunk_size=None,
    first_frame=False, forward_only=False, num_init_cond_frames=1,
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

    Returns:
        List of segmentation arrays (one per iteration), each of shape (Z, H, W).
    """
    raw_proc = pad_to_square(raw)
    labels_proc = pad_to_square(labels)

    volume_embeddings = _embedding_tensors_to_numpy(
        precompute_image_embeddings(predictor=predictor, input_=raw_proc, ndim=3)
    )
    inference_state = predictor.init_state(volume=raw_proc, volume_embeddings=volume_embeddings)

    prompt_generator = IterativePromptGenerator()
    gt_ids = sorted(np.unique(labels_proc)[1:].tolist())

    seg_per_iter_proc = [np.zeros_like(labels_proc) for _ in range(n_iterations)]

    for obj_id in tqdm(gt_ids, desc="Objects", leave=False):
        gt_3d = labels_proc == obj_id
        obj_zs = np.where(gt_3d.any(axis=(1, 2)))[0]

        if first_frame:
            z_prompt = int(obj_zs.min())
        else:
            z_prompt = int(np.ceil(np.mean([obj_zs.min(), obj_zs.max()])))

        gt_slice = (labels_proc[z_prompt] == obj_id).astype("uint32")
        points, point_labels, boxes = _get_batched_prompts(
            gt=gt_slice, gt_ids=[1],
            use_points=not use_box, use_boxes=use_box,
            n_positives=0 if use_box else 1, n_negatives=0, dilation=5,
        )

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
                    gt_extra = (labels_proc[z_extra] == obj_id).astype("uint32")
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
                    predictor, inference_state, z_prompt, labels_proc.shape[0], chunk_size,
                    forward_only=forward_only,
                )
            else:
                video_segments = {}
                for out_frame, _, out_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=z_prompt
                ):
                    video_segments[out_frame] = (out_logits[0] > 0).cpu().numpy().squeeze()

                if not forward_only and len(video_segments) < labels_proc.shape[0]:
                    for out_frame, _, out_logits in predictor.propagate_in_video(
                        inference_state, start_frame_idx=z_prompt, reverse=True
                    ):
                        if out_frame not in video_segments:
                            video_segments[out_frame] = (out_logits[0] > 0).cpu().numpy().squeeze()

            seg_3d = np.zeros(labels_proc.shape, dtype=bool)
            H_proc, W_proc = labels_proc.shape[1], labels_proc.shape[2]
            for z, mask in video_segments.items():
                seg_3d[z] = mask[:H_proc, :W_proc]

            seg_per_iter_proc[iteration][seg_3d] = obj_id

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

    # Strip square padding before evaluating against the original label dimensions.
    _, H_orig, W_orig = labels.shape
    seg_per_iter = [s[:, :H_orig, :W_orig] for s in seg_per_iter_proc]
    return seg_per_iter


@torch.no_grad()
def segment_image_2d(raw, labels, predictor, n_iterations, use_box=False, batch_size=32):
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


def evaluate_volume(labels, seg_per_iter):
    """Compute per-iteration segmentation accuracy metrics.

    Args:
        labels: Ground-truth label array.
        seg_per_iter: List of segmentation arrays, one per iteration.

    Returns:
        List of dicts with 'iteration', 'mSA', 'SA50', 'SA75'.
    """
    rows = []
    for i, seg in enumerate(seg_per_iter):
        msa, sa = mean_segmentation_accuracy(seg, labels, return_accuracies=True)
        rows.append({"iteration": i, "mSA": round(msa, 4), "SA50": round(sa[0], 4), "SA75": round(sa[5], 4)})
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
            return SAM2ImagePredictor(video_pred)
    model = get_sam2_model(model_type=model_type, backbone=backbone, checkpoint_path=checkpoint_path)
    return SAM2ImagePredictor(model)


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
            num_init_cond_frames=args.num_init_cond_frames,
        )
        rows = evaluate_volume(labels, seg_per_iter)
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
            use_box=args.prompt == "box",
        )
        rows = evaluate_volume(labels, seg_per_iter)
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

    # WIP
    # 5. Iterative rectification design (TODO: Investigate closely)
    #   a. Current: put mask / box / point in the "first frame" -> randomly rectifies n frames in one go with n points -> that's it.  # noqa
    #   b. Expectation: iterate over n times, which is controllable.

    # UPCOMING CONSIDERATIONS
    # 4. Propagate over chunks (fairly critical).
    
    # 6. TODO: They propagate the masks further externally from previous predictions!

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
    parser.add_argument(
        "--num_init_cond_frames", type=int, default=1,
        help="Frames to condition on before first propagation (1 = prompt only; 2 matches training).",
    )
    parser.add_argument(
        "--dataset", choices=["skull", "lucchi", "livecell"], default=None,
        help="Dataset to evaluate (default: all three sequentially).",
    )
    parser.add_argument("--n_eval_images", type=int, default=20, help="Number of LIVECell test images to evaluate.")
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


if __name__ == "__main__":
    main()
