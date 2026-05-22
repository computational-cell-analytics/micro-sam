"""Evaluate iterative 3D interactive segmentation on Mouse-Skull-Nuclei-CBG test data.

Results (X2_right.tif, 45 objects):

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
"""

import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from skimage.measure import label as connected_components

import torch

from torch_em.transform.raw import normalize
from torch_em.util.segmentation import size_filter

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.prompt_generators import IterativePromptGenerator
from micro_sam.evaluation.inference import _get_batched_prompts, _get_batched_iterative_prompts

from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings
from micro_sam.v2.evaluation.inference import _embedding_tensors_to_numpy


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/embedseg/Mouse-Skull-Nuclei-CBG/test"


def load_test_volumes(min_size=10):
    img_dir = os.path.join(DATA_ROOT, "images")
    mask_dir = os.path.join(DATA_ROOT, "masks")
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
        first_frame: Prompt from the first z-frame of each object (vs center).
        forward_only: Propagate forward only from the prompt frame.
        num_init_cond_frames: Number of frames to condition on before first propagation.
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


def evaluate_volume(labels, seg_per_iter):
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


def main():
    # NOTE:
    # 1. Normalization works fine, as expected, i.e. [0, 1] -> normalize with ImageNet stats
    # 2. First frame propagation working in forward direction only
    #   a. TODO: Change the training design for the model to work for bidirectional propagation?
    # 3. Resizing image, final design: resizelongestside(1024) -> pad rest to 1024 in both axes.
    #   a. TODO: This one was / is a huge mess. I need to be a bit more careful and do this consistently.
    # 4. Propagate over chunks (fairly critical).
    # 5. Iterative rectification design (TODO: Investigate closely)
    #   a. Current: put mask / box / point in the "first frame" -> randomly rectifies n frames in one go with n points -> that's it.  # noqa
    #   b. Expectation: iterate over n times, which is controllable.
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
    args = parser.parse_args()

    predictor = get_predictor(
        model_type=args.model_type,
        backbone=args.backbone,
        checkpoint_path=args.checkpoint,
    )
    predictor.add_all_frames_to_correct_as_cond = True

    samples = load_test_volumes(min_size=args.min_size)

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
        print("\nMean across volumes:")
        df_all = pd.DataFrame(all_rows)
        print(df_all.groupby("iteration")[["mSA", "SA50", "SA75"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
