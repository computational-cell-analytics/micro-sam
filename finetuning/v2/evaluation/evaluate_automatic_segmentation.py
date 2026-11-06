"""Evaluation of micro-sam2 automatic segmentation, for 2d and 3d, LM and EM.

Two modes share one pipeline:
  ais: the jointly finetuned UniSAM2 decoder, turned into instances by the flow / multicut postprocessing.
  apg: the same decoder proposes candidates, which the interactive branch turns into masks.

The parameters are tuned separately, once per dataset and mode, by `parameter_search.py`. This
script only reads what that sweep found (see `common.read_tuned_params`) and scores the test split
with it. A dataset with no data held out from the evaluation keeps the library defaults, because a
sweep there would select its parameters on the very samples the reported score is measured on. See
`common.VAL_SPLITS` for what counts as held out.

Usage examples:
    python evaluate_automatic_segmentation.py -d livecell -m hvit_b -e <exp> --mode ais
    python evaluate_automatic_segmentation.py -d gonuclear -m hvit_b -e <exp> --mode apg
    python evaluate_automatic_segmentation.py -d cremi -m hvit_b -e <exp> --mode ais --skip_tuning
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd

import torch

from micro_sam.v2.util import FINETUNED_MODELS, has_registered_decoder

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASET_SPACING, MODEL_TYPES, MODES, VOLUME_SPEED_OPTIONS, build_model,
    check_data_download, evaluate_samples, has_val_split, load_apg_overrides, postprocess_unisam2, predict_unisam2,
    pad_to_min_shape, read_tuned_params, resolve_checkpoint_identity,
)

# The 2d datasets whose test images are smaller than their training images at the same pixel size. They are
# padded with zeros to the training shape, so that micro-sam2 sees the objects at the scale it was trained on:
# the TissueNet test images are 256 x 256 and its training images 512 x 512.
TRAINING_SCALE_MIN_SHAPE_2D = {"tissuenet": (512, 512)}

# EVAL_NORM_PERCENTILES ('lower,upper', e.g. '1,99.8'): experiment that replaces the 2nd / 98th percentile bounds
# of the micro-sam2 input normalization; such runs belong in their own experiment folder.
if os.environ.get("EVAL_NORM_PERCENTILES"):
    import micro_sam.v2.normalization as _normalization
    _lower, _upper = (float(value) for value in os.environ["EVAL_NORM_PERCENTILES"].split(","))
    for _function in (_normalization.compute_percentile_bounds, _normalization.normalize_raw):
        _names = _function.__code__.co_varnames[:_function.__code__.co_argcount]
        _defaults = list(_function.__defaults__)
        _first = len(_names) - len(_defaults)
        _defaults[_names.index("lower_percentile") - _first] = _lower
        _defaults[_names.index("upper_percentile") - _first] = _upper
        _function.__defaults__ = tuple(_defaults)

REGISTRY_MODELS = [model for model in FINETUNED_MODELS if has_registered_decoder(model)]

# The temporary embeddings of APG go to the job's node-local TMPDIR, as in `parameter_search.py`.
# In the micro-sam cache of the home directory, the leftovers of killed jobs fill the quota.
if os.environ.get("TMPDIR"):
    import uuid
    import atexit
    import shutil
    from pathlib import Path
    import micro_sam.util
    import micro_sam.v2.util
    import micro_sam.v2.instance_segmentation
    import micro_sam.v2.automatic_prompt_generation

    def _make_temp_embedding_path():
        parent = Path(os.environ["TMPDIR"], "micro_sam_tmp_embeddings")
        parent.mkdir(parents=True, exist_ok=True)
        path = str(parent / f"{uuid.uuid4().hex}.zarr")
        atexit.register(shutil.rmtree, path, ignore_errors=True)
        return path

    for _module in (micro_sam.util, micro_sam.v2.util, micro_sam.v2.instance_segmentation,
                    micro_sam.v2.automatic_prompt_generation):
        _module.make_temp_embedding_path = _make_temp_embedding_path


def segment(
    model, mode, raw, ndim, dataset_name, model_type, params, device, spacing=None, devices=None, min_shape=None,
):
    """Segment one sample with the tuned parameters of a mode.

    With 'min_shape' (y, x), a 2d image smaller than it is padded with zeros to that shape, as PathoSAM does with the
    256 x 256 PanNuke tiles, and the segmentation is cropped back. A volume is padded in y and x, slice by slice.
    """
    if min_shape is not None and ndim == 2 and min(raw.shape[:2]) < min(min_shape):
        from skimage.segmentation import relabel_sequential
        padded, crop = pad_to_min_shape(raw, min_shape)
        seg = segment(model, mode, padded, ndim, dataset_name, model_type, params, device, spacing, devices)
        return relabel_sequential(seg[crop])[0].astype("uint32")
    if min_shape is not None and ndim == 3 and min(raw.shape[1:3]) < min(min_shape):
        from skimage.segmentation import relabel_sequential
        # The z axis is moved last, so that pad_to_min_shape pads the y and x axes of every slice.
        padded, crop = pad_to_min_shape(np.moveaxis(raw, 0, -1), min_shape)
        seg = segment(model, mode, np.moveaxis(padded, -1, 0), ndim, dataset_name, model_type, params, device, spacing,
                      devices)
        return relabel_sequential(seg[(slice(None),) + crop])[0].astype("uint32")
    if mode == "apg":
        model.clear_state()
        model.initialize(raw, ndim=ndim, **(VOLUME_SPEED_OPTIONS if ndim == 3 else {}))
        volume_params = {"spacing": spacing} if ndim == 3 else {}
        return model.generate(**{**volume_params, **params}).astype("uint32")

    prediction = predict_unisam2(model, raw, ndim=ndim, device=device, devices=devices)
    return postprocess_unisam2(prediction, dataset_name, model_type=model_type, params=params)


def run_evaluation(
    model, mode, dataset_name, data_root, experiment_folder, model_type, params, device, limit,
    crop_shape=None, checkpoint_id=None, devices=None, tuned=None, result_tag=None, config_name=None, sample_index=None,
    shard=None, min_shape=None,
):
    """Score the test split with the given parameters and write the result CSV.

    The parameters land in the result file next to the metrics, so a number can always be traced back
    to the run that produced it.

    Args:
        model: The model of the mode, from `common.build_model`.
        mode: The segmentation mode, one of MODES.
        dataset_name: The dataset to score.
        data_root: The root the data lives in.
        experiment_folder: The folder the results are written to.
        model_type: The model, e.g. 'hvit_t' or 'hvit_s_em_mito', which names the result file.
        params: The parameters to segment with, or None for the library defaults.
        device: The torch device.
        limit: Score only the first this many samples, for a smoke test. The result file is named
            after it, so a truncated run cannot be mistaken for the full evaluation.
        crop_shape: The 3d center crop.
        checkpoint_id: The checksum of all model weights used by the mode.
        devices: The devices inference spreads over. All visible GPUs by default.
        tuned: Whether 'params' came from the tuning sweep. Names the result file 'tuned' or
            'default'; by default inferred from whether there are parameters at all.
        result_tag: Optional tag appended to the result file name, so that a run with explicit
            parameter overrides does not collide with the plain evaluation.
        config_name: The name of the configuration the overrides came from, stored in the results.
        sample_index: The index of the only sample to score, for one array task. See `common.evaluate_samples`.
        shard: (shard_index, n_shards) to score one stride of the samples, for one array task.

    Returns:
        The results as a DataFrame, or None while the rows of other samples are missing.
    """
    if tuned is None:
        tuned = bool(params)
    tag = "tuned" if tuned else "default"
    if result_tag:
        tag = f"{tag}_{result_tag}"
    if limit is not None:
        tag = f"{tag}_n{limit}"
    legacy_path = os.path.join(
        experiment_folder, "results", f"{dataset_name}_micro_sam2_{model_type}_{mode}_{tag}.csv"
    )
    save_path = legacy_path if checkpoint_id is None else legacy_path[:-4] + f"_ckpt-{checkpoint_id}.csv"
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return pd.read_csv(save_path)
    if checkpoint_id is not None and os.path.exists(legacy_path):
        warnings.warn(
            f"Using legacy evaluation result '{legacy_path}'. It has no checkpoint checksum, so its "
            "weights cannot be verified."
        )
        return pd.read_csv(legacy_path)

    ndim = 3 if dataset_name in DATASETS_3D else 2
    spacing = DATASET_SPACING.get(dataset_name)
    extra_columns = {"parameters": json.dumps(params, sort_keys=True, default=str) if params else "default"}
    if config_name is not None:
        extra_columns["config_name"] = config_name
    return evaluate_samples(
        lambda raw: segment(
            model, mode, raw, ndim, dataset_name, model_type, params or {}, device, spacing=spacing, devices=devices,
            min_shape=min_shape,
        ),
        dataset_name, data_root, save_path, desc=f"{mode}-{model_type}", limit=limit, crop_shape=crop_shape,
        sample_index=sample_index, extra_columns=extra_columns, shard=shard,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", required=True, choices=sorted(DATASETS_2D + DATASETS_3D))
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument(
        "-m", "--model_type", type=str, default="hvit_t", choices=MODEL_TYPES + REGISTRY_MODELS,
        help="A backbone, or a registry model whose family selects the default parameters, e.g. 'hvit_s_em_mito'.",
    )
    parser.add_argument("--mode", type=str, default="ais", choices=MODES, help="The segmentation mode to evaluate.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Weights instead of the joint export.")
    parser.add_argument("--joint_checkpoint", type=str, default="best", help="Joint checkpoint name, without '.pt'.")
    parser.add_argument(
        "--interactive_checkpoint", type=str, default=None,
        help="Standalone interactive weights for --mode apg, bypassing the joint checkpoint entirely. "
             "Requires -c/--checkpoint for the decoder half too.",
    )
    parser.add_argument("--skip_tuning", action="store_true", help="Evaluate with the library defaults.")
    parser.add_argument("--n_samples", type=int, default=None, help="Score only the first N samples, for a check.")
    parser.add_argument("--sample_index", type=int, default=None, help="Score only this sample, as one array task.")
    parser.add_argument("--n_shards", type=int, default=None, help="Split the samples over this many array tasks.")
    parser.add_argument("--shard_index", type=int, default=None, help="The shard this array task scores.")
    parser.add_argument("--tuning_root", type=str, default=None, help="Where parameter_search.py wrote its sweeps.")
    parser.add_argument("--crop_3d", type=int, nargs=3, default=None, help="Override the 3d crop (Z Y X).")
    parser.add_argument(
        "--propagation_waves", type=int, default=None,
        help="Volumes only. Rounds the candidates are propagated in, overriding whatever was tuned.",
    )
    parser.add_argument("--devices", nargs="*", default=None, help="Inference devices. All visible GPUs by default.")
    parser.add_argument(
        "--apg_params", type=str, default=None,
        help="APG only. A JSON configuration in the benchmark format. Its section for the dataset ('params_2d', "
        "'params_3d' or 'params_dense') overrides the tuned parameters, or the defaults with --skip_tuning.",
    )
    parser.add_argument(
        "--result_tag", type=str, default=None,
        help="Tag appended to the result file name. Defaults to the --apg_params configuration name.",
    )
    parser.add_argument(
        "--pad_to_min_shape", type=int, nargs=2, default=None,
        help="Pad images (or the slices of a volume) smaller than this (Y X) with zeros and crop the segmentation "
        "back. Defaults to TRAINING_SCALE_MIN_SHAPE_2D of the dataset.",
    )
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)
    if args.apg_params is not None and args.mode != "apg":
        parser.error("--apg_params applies to --mode apg only.")

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ndim = 3 if args.dataset_name in DATASETS_3D else 2
    crop_shape = tuple(args.crop_3d) if args.crop_3d else None
    checkpoint_id, joint_checksum = resolve_checkpoint_identity(
        args.mode, args.model_type, args.joint_checkpoint, args.checkpoint,
        interactive_checkpoint_path=args.interactive_checkpoint,
    )
    model = build_model(
        args.mode, args.model_type, device, ndim,
        joint_checkpoint=args.joint_checkpoint, checkpoint_path=args.checkpoint,
        joint_checksum=joint_checksum, interactive_checkpoint_path=args.interactive_checkpoint,
        devices=args.devices or None,
    )
    params = None
    tuned = False
    if not args.skip_tuning:
        if has_val_split(args.dataset_name):
            tuning_root = args.tuning_root or os.path.join(args.experiment_folder, "tuning")
            tuning_root = os.path.join(tuning_root, args.mode)
            try:
                params = read_tuned_params(tuning_root, args.dataset_name, args.model_type[:6], checkpoint_id)
                tuned = True
            except FileNotFoundError:
                warnings.warn(
                    f"No tuned parameters for '{args.dataset_name}' under '{tuning_root}'. Run "
                    "parameter_search.py first, or pass --skip_tuning. Using the library defaults."
                )
        else:
            print(f"'{args.dataset_name}' has no data held out from the evaluation, so the defaults are used.")

    if args.propagation_waves is not None:
        # On top of the tuned set, or on its own: a volume takes it either way, and the defaults for
        # everything else are what 'generate' applies when a key is missing.
        params = {**(params or {}), "propagation_waves": args.propagation_waves}

    config_name, result_tag = None, args.result_tag
    if args.apg_params is not None:
        config_name, overrides = load_apg_overrides(args.apg_params, args.dataset_name)
        params = {**(params or {}), **overrides}
        if result_tag is None:
            result_tag = config_name

    run_evaluation(
        model, args.mode, args.dataset_name, args.input_path, args.experiment_folder, args.model_type,
        params, device, crop_shape=crop_shape, checkpoint_id=checkpoint_id,
        devices=args.devices or None, tuned=tuned, result_tag=result_tag, config_name=config_name,
        limit=args.n_samples, sample_index=args.sample_index,
        shard=None if args.n_shards is None else (args.shard_index, args.n_shards),
        min_shape=(
            TRAINING_SCALE_MIN_SHAPE_2D.get(args.dataset_name) if args.pad_to_min_shape is None
            else tuple(args.pad_to_min_shape)
        ),
    )


if __name__ == "__main__":
    main()
