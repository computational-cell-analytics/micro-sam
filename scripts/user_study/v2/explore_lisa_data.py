"""Automatic instance segmentation with APG or AIS and 'hvit_t_cells' on a crop of the lisa_data RGYB image.

The 'tissuenet_norm' input takes channel 1 (R) as channel 1, channel 4 (B) as channel 2 and sets channel 3 to zeros.
The 'channel_1' input puts channel 1 into all three channels, as micro-sam does for a single-channel image.
Every channel is normalized TissueNet style, by its own 2nd and 98th percentile over the crop. The decoder outputs
(--predictions) compare this with other normalizations, each passed to the encoder as the [0, 1] image.
"""

import os
import time
import argparse
from glob import glob
from contextlib import contextmanager

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries

import torch

from elf.io import open_file

from bioimage_cpp.segmentation import label, watershed

from micro_sam.v2 import batched_inference
from micro_sam.v2.util import encode_image
from micro_sam.v2.transforms.resize import ResizeLongestSideAndPad
from micro_sam.v2.normalization import compute_percentile_bounds, normalize_raw, to_image
from micro_sam.v2.postprocessing import _compute_flow_density, default_postprocessing, watershed_heightmap
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter


IMAGE_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/lisa_data/Restain_Day7_2_S190131_RGYB.tiff"
OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/lisa_data"
TISSUENET_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/tissuenet/train"
N_TISSUENET = 200
NUCLEUS_GAINS = (1.0, 0.75, 0.5, 0.25)
# The apparent object scale: a region of CROP_SIZE / scale pixels is downsampled to CROP_SIZE.
SCALES = (1.0, 0.75, 0.5, 0.33)
ENGINES = ("apg", "ais")
PLOT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_TYPE = "hvit_t_cells"
CENTER = (4800, 9800)  # (y, x)
CROP_SIZE = 512
OVERVIEW_STEP = 16
NAME = f"Restain_Day7_2_S190131_crop_y{CENTER[0]}_x{CENTER[1]}"
# The 0-based file channel of each of the three model input channels, None for zeros.
INPUT_CHANNELS = {"tissuenet_norm": (0, 3, None), "channel_1": (0, 0, 0)}
# The lower and upper percentile of each input normalization, applied per channel.
NORMALIZATIONS = {"minmax": (0, 100), "p1_p99": (1, 99), "p2_p98": (2, 98)}
PREDICTION_CHANNELS = ("Foreground", "Z distance (unused in 2d)", "Y distance", "X distance")
# The AIS sparse post-processing parameters to sweep, each scaled from its default by the factors.
SWEEP_PARAMETERS = ("density_threshold", "sigma", "n_iter", "min_size")
SWEEP_FACTORS = (1, 2, 4)
# 384 + 2 * 64 = 512 px per SAM input, the same 2x upsampling as the validated 512 crop.
LARGE_TILE_SHAPE, LARGE_HALO = (384, 384), (64, 64)
# Objects whose mean raw channel 1 value is below this sit on dark background (tissue averages about 34).
DARK_THRESHOLD = 10
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_crop(size):
    y, x = (slice(center - size // 2, center - size // 2 + size) for center in CENTER)
    return np.moveaxis(tifffile.memmap(IMAGE_PATH)[:, y, x], 0, -1)  # (C, Y, X) -> (Y, X, C)


def load_data():
    overview = tifffile.memmap(IMAGE_PATH)[:, ::OVERVIEW_STEP, ::OVERVIEW_STEP]
    return load_crop(CROP_SIZE), np.moveaxis(overview, 0, -1)


def to_rgb_display(image):
    return to_image(image[..., [0, 1, 3]])


def describe_channels(input_name):
    return ", ".join("zeros" if source is None else f"channel {source + 1}" for source in INPUT_CHANNELS[input_name])


def to_model_input(crop, input_name, percentiles=(2, 98)):
    lower, upper = percentiles
    return np.stack([
        np.zeros(crop.shape[:2], dtype="float32") if source is None
        else normalize_raw(crop[..., source], lower_percentile=lower, upper_percentile=upper)
        for source in INPUT_CHANNELS[input_name]
    ], axis=-1)


def print_intensities(crop, image, input_name):
    for c, source in enumerate(INPUT_CHANNELS[input_name]):
        normalized = image[..., c]
        if source is None:
            print(f"{input_name}, model input {c}: zeros, max {normalized.max()}")
            continue
        raw = crop[..., source]
        lower, upper = compute_percentile_bounds(raw)
        print(
            f"{input_name}, model input {c} (channel {source + 1}): raw {raw.dtype} {raw.min()}-{raw.max()}, "
            f"p2 {lower.item():.1f}, p98 {upper.item():.1f}, "
            f"normalized {normalized.dtype} {normalized.min():.2f}-{normalized.max():.2f}, "
            f"at 0: {(normalized == 0).mean():.3f}, at 1: {(normalized == 1).mean():.3f}"
        )


def random_label_colors(n_labels, seed=42):
    colors = np.random.default_rng(seed).uniform(0.2, 1.0, size=(n_labels + 1, 3))
    colors[0] = 0
    return colors


def save_figure(fig, name):
    plot_path = os.path.join(PLOT_ROOT, f"{name}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved the plot to {plot_path}")


def plot_crop(crop, overview, images):
    fig, axes = plt.subplots(1, 2 + len(images), figsize=(6.7 * (2 + len(images)), 7))
    axes[0].imshow(to_rgb_display(overview))
    corner = [(center - CROP_SIZE // 2) / OVERVIEW_STEP for center in CENTER]
    size = CROP_SIZE / OVERVIEW_STEP
    axes[0].add_patch(Rectangle(corner[::-1], size, size, fill=False, edgecolor="yellow", linewidth=2))
    axes[0].set_title(f"Whole image, crop at (y, x) = {CENTER}")
    axes[1].imshow(to_rgb_display(crop))
    axes[1].set_title("Crop: channels 1, 2, 4 as RGB")
    for ax, (input_name, image) in zip(axes[2:], images.items()):
        ax.imshow(to_image(image))
        ax.set_title(f"Model input: {describe_channels(input_name)}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{NAME} ({CROP_SIZE}x{CROP_SIZE})", y=1.02)
    fig.tight_layout()
    save_figure(fig, NAME)


def plot_result(crop, image, segmentation, method, input_name):
    display = to_image(image)
    overlay = display.copy()
    overlay[find_boundaries(segmentation, mode="inner")] = (255, 255, 0)

    label = method.upper().replace("_", " ")
    n_objects = len(np.unique(segmentation[segmentation > 0]))
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))
    panels = [
        (to_rgb_display(crop), "Crop: channels 1, 2, 4 as RGB"),
        (display, f"Model input: {describe_channels(input_name)}"),
        (random_label_colors(int(segmentation.max()))[segmentation], f"{label}: {n_objects} objects"),
        (overlay, f"{label} boundaries on the model input"),
    ]
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{NAME} ({MODEL_TYPE})", y=1.02)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_{method}_{input_name}")


def capture_encoder_inputs(predictor):
    inputs = []
    encoder = getattr(predictor, "model", predictor).image_encoder
    encoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().float().cpu()))
    return inputs


def check_encoder_inputs(inputs, image):
    mean, std = (torch.tensor(stats)[:, None, None] for stats in (IMAGENET_MEAN, IMAGENET_STD))
    expected = ResizeLongestSideAndPad(inputs[0].shape[-1])(torch.from_numpy(image).permute(2, 0, 1))
    for i, encoder_input in enumerate(inputs):
        encoder_input = encoder_input[0]
        unit_range = encoder_input * std + mean
        print(f"Encoder call {i}: input {tuple(encoder_input.shape)}")
        for c in range(3):
            print(
                f"Channel {c}: encoder input {encoder_input[c].min():.3f} to {encoder_input[c].max():.3f}, "
                f"without the ImageNet normalization {unit_range[c].min():.3f} to {unit_range[c].max():.3f}"
            )
        difference = (unit_range - expected).abs().max()
        print(f"Max difference to the resized [0, 1] input: {difference:.5f} ({difference * 255:.2f} uint8 steps)")


def run_segmentation(crop, images, engine, modes, check_encoder_input, density_threshold):
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine)
    encoder_inputs = capture_encoder_inputs(predictor) if check_encoder_input else None
    methods = {f"ais_{mode}": mode for mode in modes} if engine == "ais" else {"apg": None}
    postprocessing = {}
    if engine == "ais" and density_threshold is not None:
        methods = {f"{method}_density_threshold_{density_threshold:g}": mode for method, mode in methods.items()}
        postprocessing["density_threshold"] = density_threshold

    for input_name, image in images.items():
        for method, mode in methods.items():
            output_path = os.path.join(OUTPUT_ROOT, f"{NAME}_{MODEL_TYPE}_{method}_{input_name}.tif")
            segmentation = automatic_instance_segmentation(
                predictor, segmenter, input_path=image, output_path=output_path, ndim=2,
                **({} if mode is None else {"mode": mode}), **postprocessing,
            )
            if check_encoder_input:
                check_encoder_inputs(encoder_inputs, image)
                encoder_inputs.clear()
            plot_result(crop, image, segmentation, method, input_name)


def encode(predictor, image):
    # The front-end would normalize by 2-98 again, so encode the [0, 1] image directly.
    encode_image(predictor, image)
    return {
        "features": predictor.get_image_embedding().cpu().numpy(),
        "high_res_feats": predictor._features["high_res_feats"],
        "input_size": predictor.model.image_size,
        "original_size": predictor._orig_hw,
    }


def predict_ais(predictor, segmenter, image):
    segmenter.clear_state()
    segmenter.initialize(image, ndim=2, image_embeddings=encode(predictor, image))
    return segmenter.get_state()["prediction"], segmenter.generate(mode="sparse")


def watershed_steps(prediction):
    # The steps of 'flow_instance_segmentation', the AIS sparse post-processing, with its defaults.
    params = default_postprocessing(MODEL_TYPE, "sparse")
    foreground, distances = prediction[0], prediction[2:]
    foreground_mask = foreground > params["foreground_threshold"]
    density = _compute_flow_density(
        distances, foreground_mask, n_iter=params["n_iter"], dt=params["dt"], sigma=params["sigma"]
    )
    seeds = label(density > params["density_threshold"])
    heightmap = watershed_heightmap(foreground, distances, params["foreground_weight"])
    segmentation = watershed(heightmap, markers=seeds, mask=foreground_mask)
    ids, sizes = np.unique(segmentation, return_counts=True)
    segmentation[np.isin(segmentation, ids[(sizes < params["min_size"]) & (ids > 0)])] = 0
    segmentation = watershed(heightmap, markers=segmentation, mask=foreground_mask)
    steps = {"foreground_mask": foreground_mask, "density": density, "seeds": seeds, "heightmap": heightmap}
    return steps, segmentation


def get_normalization_predictions(crop, input_name, normalizations, check_encoder_input):
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais")
    encoder_inputs = capture_encoder_inputs(predictor) if check_encoder_input else None
    results = {}
    for normalization in normalizations:
        image = to_model_input(crop, input_name, NORMALIZATIONS[normalization])
        prediction, segmentation = predict_ais(predictor, segmenter, image)
        steps, watershed_segmentation = watershed_steps(prediction)
        assert np.array_equal(watershed_segmentation, segmentation)
        if check_encoder_input:
            check_encoder_inputs(encoder_inputs, image)
            encoder_inputs.clear()

        output_path = os.path.join(OUTPUT_ROOT, f"{NAME}_{MODEL_TYPE}_ais_predictions_{input_name}_{normalization}.tif")
        tifffile.imwrite(output_path, prediction)
        output_path = os.path.join(OUTPUT_ROOT, f"{NAME}_{MODEL_TYPE}_ais_sparse_{input_name}_{normalization}.tif")
        tifffile.imwrite(output_path, segmentation, compression="zlib")
        n_seeds = len(np.unique(steps["seeds"][steps["seeds"] > 0]))
        n_objects = len(np.unique(segmentation[segmentation > 0]))
        print(f"{normalization}: {n_seeds} seeds, {n_objects} objects with AIS sparse")
        for c, source in enumerate(INPUT_CHANNELS[input_name]):
            if source is not None:
                channel = image[..., c]
                saturated = (channel == 1).mean()
                print(f"{normalization}, channel {source + 1}: mean {channel.mean():.3f}, at 1: {saturated:.3f}")
        for channel_name, channel in zip(PREDICTION_CHANNELS, prediction):
            print(f"{normalization}, {channel_name}: {channel.min():.3f} to {channel.max():.3f}")
        results[normalization] = (image, prediction, segmentation, steps)
    return results


def plot_normalizations(input_name, results):
    # Shared color limits per distance channel, so the rows compare directly.
    limits = [max(np.abs(prediction[c]).max() for _, prediction, _, _ in results.values()) for c in range(1, 4)]
    fig, axes = plt.subplots(len(results), 5, figsize=(31, 6 * len(results)), squeeze=False)
    for row, (normalization, (image, prediction, segmentation, _)) in zip(axes, results.items()):
        n_objects = len(np.unique(segmentation[segmentation > 0]))
        row[0].imshow(np.clip(image, 0, 1))
        row[0].set_title(f"{normalization}: {describe_channels(input_name)} (AIS sparse: {n_objects} objects)")
        plot = row[1].imshow(prediction[0], cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(plot, ax=row[1], fraction=0.046, pad=0.04)
        row[1].set_title(PREDICTION_CHANNELS[0])
        for ax, channel_name, channel, limit in zip(row[2:], PREDICTION_CHANNELS[1:], prediction[1:], limits):
            plot = ax.imshow(channel, cmap="RdBu_r", vmin=-limit, vmax=limit)
            fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(channel_name)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{NAME} ({MODEL_TYPE}): AIS decoder outputs per input normalization", y=1.01)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_ais_predictions_{input_name}")


def plot_watershed(input_name, results):
    density_limit = max(np.percentile(steps["density"], 99.5) for _, _, _, steps in results.values())
    fig, axes = plt.subplots(len(results), 5, figsize=(31, 6 * len(results)), squeeze=False)
    for row, (normalization, (image, _, segmentation, steps)) in zip(axes, results.items()):
        display = (np.clip(image, 0, 1) * 255).astype("uint8")
        n_seeds = len(np.unique(steps["seeds"][steps["seeds"] > 0]))
        n_objects = len(np.unique(segmentation[segmentation > 0]))

        row[0].imshow(display)
        row[0].set_title(f"{normalization}: {describe_channels(input_name)}")
        row[1].imshow(steps["foreground_mask"], cmap="gray")
        row[1].set_title(f"Foreground mask ({steps['foreground_mask'].mean():.2f} of the crop)")
        plot = row[2].imshow(steps["density"], cmap="magma", vmin=0, vmax=density_limit)
        fig.colorbar(plot, ax=row[2], fraction=0.046, pad=0.04)
        row[2].contour(steps["seeds"] > 0, levels=[0.5], colors="cyan", linewidths=0.8)
        row[2].set_title(f"Flow density, seeds in cyan ({n_seeds} seeds)")
        plot = row[3].imshow(steps["heightmap"], cmap="viridis")
        fig.colorbar(plot, ax=row[3], fraction=0.046, pad=0.04)
        row[3].set_title("Watershed heightmap")
        overlay = display.copy()
        overlay[find_boundaries(segmentation, mode="inner")] = (255, 255, 0)
        row[4].imshow(overlay)
        row[4].set_title(f"Watershed: {n_objects} objects")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{NAME} ({MODEL_TYPE}): AIS sparse watershed per input normalization", y=1.01)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_ais_watershed_{input_name}")


def segment_apg(predictor, segmenter, image):
    segmenter.initialize(image, ndim=2, image_embeddings=encode(predictor, image))
    return segmenter.generate()


def run_apg_normalizations(crop, input_name, normalizations, check_encoder_input):
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="apg")
    encoder_inputs = capture_encoder_inputs(predictor) if check_encoder_input else None
    results = {}
    for normalization in normalizations:
        image = to_model_input(crop, input_name, NORMALIZATIONS[normalization])
        segmentation = segment_apg(predictor, segmenter, image)
        if check_encoder_input:
            check_encoder_inputs(encoder_inputs, image)
            encoder_inputs.clear()

        output_path = os.path.join(OUTPUT_ROOT, f"{NAME}_{MODEL_TYPE}_apg_{input_name}_{normalization}.tif")
        tifffile.imwrite(output_path, segmentation, compression="zlib")
        print(f"{normalization}: {len(np.unique(segmentation[segmentation > 0]))} objects with APG")
        results[normalization] = (image, segmentation)
    return results


def plot_apg_normalizations(input_name, results):
    fig, axes = plt.subplots(len(results), 3, figsize=(19, 6.5 * len(results)), squeeze=False)
    for row, (normalization, (image, segmentation)) in zip(axes, results.items()):
        display = (np.clip(image, 0, 1) * 255).astype("uint8")
        overlay = display.copy()
        overlay[find_boundaries(segmentation, mode="inner")] = (255, 255, 0)
        n_objects = len(np.unique(segmentation[segmentation > 0]))
        row[0].imshow(display)
        row[0].set_title(f"{normalization}: {describe_channels(input_name)}")
        row[1].imshow(random_label_colors(int(segmentation.max()))[segmentation])
        row[1].set_title(f"APG: {n_objects} objects")
        row[2].imshow(overlay)
        row[2].set_title("APG boundaries on the model input")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{NAME} ({MODEL_TYPE}): APG per input normalization", y=1.01)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_apg_{input_name}_normalizations")


def sweep_postprocessing(crop, input_name):
    image = to_model_input(crop, input_name)
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais")
    automatic_instance_segmentation(predictor, segmenter, input_path=image, ndim=2, verbose=False)

    defaults = default_postprocessing(MODEL_TYPE, "sparse")
    results = {}
    for parameter in SWEEP_PARAMETERS:
        for factor in SWEEP_FACTORS:
            value = type(defaults[parameter])(defaults[parameter] * factor)
            segmentation = segmenter.generate(mode="sparse", **{parameter: value})
            print(f"{parameter} = {value}: {len(np.unique(segmentation[segmentation > 0]))} objects")
            results[(parameter, value, factor == 1)] = segmentation
    return image, results


def plot_sweep(input_name, image, results):
    display = to_image(image)
    fig, axes = plt.subplots(len(SWEEP_PARAMETERS), len(SWEEP_FACTORS), figsize=(19, 6.5 * len(SWEEP_PARAMETERS)))
    for ax, ((parameter, value, is_default), segmentation) in zip(axes.flat, results.items()):
        overlay = display.copy()
        overlay[find_boundaries(segmentation, mode="inner")] = (255, 255, 0)
        n_objects = len(np.unique(segmentation[segmentation > 0]))
        ax.imshow(overlay)
        ax.set_title(f"{parameter} = {value}{' (default)' if is_default else ''}: {n_objects} objects")
        ax.axis("off")
    fig.suptitle(f"{NAME} ({MODEL_TYPE}): AIS sparse post-processing, {describe_channels(input_name)}", y=1.01)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_ais_sparse_sweep_{input_name}")


def nucleus_fractions(cells, nucleus_mask):
    inside = cells > 0
    areas = np.bincount(cells[inside])
    nucleus_areas = np.bincount(cells[inside & nucleus_mask], minlength=len(areas))
    ids = np.flatnonzero(areas)
    return ids, nucleus_areas[ids] / areas[ids], areas[ids]


def input_statistics(image, cells):
    membrane, nucleus = image[..., 0], image[..., 1]
    nucleus_mask, inside = nucleus > 0.5, cells > 0
    _, fractions, areas = nucleus_fractions(cells, nucleus_mask)
    return {
        "membrane mean in cells": membrane[inside].mean(),
        "nucleus mean in nuclei": nucleus[nucleus_mask].mean(),
        "membrane mean in nuclei": membrane[nucleus_mask].mean(),
        "nucleus area / cell area": nucleus_mask[inside].mean(),
        "cells": fractions.size,
        "median cell area": np.median(areas),
        "cells > 50% nucleus": (fractions > 0.5).mean(),
    }


def compare_tissuenet(crop):
    image = to_model_input(crop, "tissuenet_norm")
    ours = {}
    for engine in ("ais", "apg"):
        predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine)
        segmentation = automatic_instance_segmentation(predictor, segmenter, input_path=image, ndim=2, verbose=False)
        ours[engine] = input_statistics(image, segmentation)

    tissuenet = tissuenet_statistics()
    print(f"Statistic: ours (AIS sparse) | ours (APG) | TissueNet median of {N_TISSUENET} images (GT cells)")
    for key in ours["ais"]:
        print(f"{key}: {ours['ais'][key]:.3f} | {ours['apg'][key]:.3f} | {tissuenet[key]:.3f}")


def tissuenet_statistics():
    statistics = []
    for path in sorted(glob(os.path.join(TISSUENET_ROOT, "*.zarr")))[:N_TISSUENET]:
        with open_file(path, "r") as f:
            rgb, cells = f["raw/rgb"][:], f["labels/cell"][:]
        # The training transform of TissueNet: per-channel 2-98 normalization of the (3, H, W) image.
        statistics.append(input_statistics(np.moveaxis(normalize_raw(rgb, axis=(1, 2)), 0, -1), cells))
    return {key: np.median([stats[key] for stats in statistics]) for key in statistics[0]}


def run_channel_gains(crop):
    base = to_model_input(crop, "tissuenet_norm")
    apg = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="apg")
    ais = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais")

    # Gains that match the channel means inside cells and nuclei to TissueNet.
    ours, tissuenet = input_statistics(base, predict_ais(*ais, base)[1]), tissuenet_statistics()
    membrane_gain = tissuenet["membrane mean in cells"] / ours["membrane mean in cells"]
    nucleus_gain = tissuenet["nucleus mean in nuclei"] / ours["nucleus mean in nuclei"]
    variants = {f"nucleus_x{gain}": (1.0, gain) for gain in NUCLEUS_GAINS}
    variants["tissuenet_matched"] = (membrane_gain, nucleus_gain)

    results = {}
    for variant, (membrane_gain, nucleus_gain) in variants.items():
        # The front-end would normalize each channel again and undo the gains, so these are encoded directly.
        image = np.clip(base * np.array([membrane_gain, nucleus_gain, 1.0], dtype="float32"), 0, 1)
        segmentations = {"apg": segment_apg(*apg, image), "ais": predict_ais(*ais, image)[1]}
        for engine, segmentation in segmentations.items():
            output_path = os.path.join(OUTPUT_ROOT, f"{NAME}_{MODEL_TYPE}_{engine}_{variant}.tif")
            tifffile.imwrite(output_path, segmentation, compression="zlib")
            _, fractions, _ = nucleus_fractions(segmentation, base[..., 1] > 0.5)
            print(
                f"{variant} (membrane x{membrane_gain:.2f}, nucleus x{nucleus_gain:.2f}), {engine}: "
                f"{fractions.size} objects, {(fractions > 0.5).mean():.3f} mostly nucleus"
            )
        results[variant] = ((membrane_gain, nucleus_gain), image, segmentations)
    return base, results


def overlay_objects(display, segmentation, nucleus_mask):
    ids, fractions, _ = nucleus_fractions(segmentation, nucleus_mask)
    nucleus_ids = ids[fractions > 0.5]
    boundaries = find_boundaries(segmentation, mode="inner")
    overlay = display.copy()
    overlay[boundaries] = (255, 255, 0)
    overlay[boundaries & np.isin(segmentation, nucleus_ids)] = (0, 255, 255)
    return overlay, len(ids), len(nucleus_ids)


def plot_channel_gains(base, results):
    nucleus_mask = base[..., 1] > 0.5
    fig, axes = plt.subplots(len(results), 3, figsize=(19, 6.5 * len(results)), squeeze=False)
    for row, (variant, (gains, image, segmentations)) in zip(axes, results.items()):
        display = (image * 255).astype("uint8")
        row[0].imshow(display)
        row[0].set_title(f"{variant}: membrane x{gains[0]:.2f}, nucleus x{gains[1]:.2f}")
        for ax, (engine, segmentation) in zip(row[1:], segmentations.items()):
            overlay, n_objects, n_nucleus = overlay_objects(display, segmentation, nucleus_mask)
            ax.imshow(overlay)
            ax.set_title(f"{engine.upper()}: {n_objects} objects, {n_nucleus} mostly nucleus (cyan)")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{NAME} ({MODEL_TYPE}): channel gains after the 2-98 normalization", y=1.005)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_channel_gains")


def load_scaled_crop(scale):
    size = int(round(CROP_SIZE / scale))
    region = load_crop(size)
    if size == CROP_SIZE:
        return size, region
    shape = (CROP_SIZE, CROP_SIZE, region.shape[-1])
    return size, resize(region.astype("float32"), shape, order=1, anti_aliasing=True, preserve_range=True)


def run_scales():
    engines = {}
    for engine in ENGINES:
        engines[engine] = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine)
    results = {}
    for scale in SCALES:
        size, crop = load_scaled_crop(scale)
        image = to_model_input(crop, "tissuenet_norm")
        segmentations = {}
        for engine, (predictor, segmenter) in engines.items():
            segmentation = automatic_instance_segmentation(predictor, segmenter, input_path=image, ndim=2)
            output_path = os.path.join(OUTPUT_ROOT, f"{NAME}_{MODEL_TYPE}_{engine}_scale_{scale}.tif")
            tifffile.imwrite(output_path, segmentation, compression="zlib")
            _, fractions, areas = nucleus_fractions(segmentation, image[..., 1] > 0.5)
            print(
                f"scale {scale} ({size} px region), {engine}: {fractions.size} objects, "
                f"{(fractions > 0.5).sum()} mostly nucleus, median area {np.median(areas):.0f} px in the input, "
                f"{np.median(areas) / scale ** 2:.0f} px at full resolution"
            )
            segmentations[engine] = segmentation
        results[scale] = (size, image, segmentations)
    return results


def plot_scales(results):
    fig, axes = plt.subplots(len(results), 3, figsize=(19, 6.5 * len(results)), squeeze=False)
    for row, (scale, (size, image, segmentations)) in zip(axes, results.items()):
        display = (image * 255).astype("uint8")
        row[0].imshow(display)
        row[0].set_title(f"scale {scale}: {size}x{size} region downsampled to {CROP_SIZE}x{CROP_SIZE}")
        for ax, (engine, segmentation) in zip(row[1:], segmentations.items()):
            overlay, n_objects, n_nucleus = overlay_objects(display, segmentation, image[..., 1] > 0.5)
            ax.imshow(overlay)
            ax.set_title(f"{engine.upper()}: {n_objects} objects, {n_nucleus} mostly nucleus (cyan)")
        if scale != 1.0:
            # The region of the scale 1.0 crop.
            corner, extent = (CROP_SIZE - CROP_SIZE * scale) / 2, CROP_SIZE * scale
            for ax in row:
                box = Rectangle((corner, corner), extent, extent, fill=False, edgecolor="white", linestyle="--")
                ax.add_patch(box)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{NAME} ({MODEL_TYPE}): apparent object scale, white box = the 512 crop", y=1.005)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_scales")


def run_crop_sizes(crop_sizes, density_threshold):
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais")
    postprocessing = {} if density_threshold is None else {"density_threshold": density_threshold}
    method = "ais_sparse" if density_threshold is None else f"ais_sparse_density_threshold_{density_threshold:g}"
    results = {}
    for size in crop_sizes:
        image = to_model_input(load_crop(size), "tissuenet_norm")
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=image, ndim=2, verbose=False, **postprocessing
        )
        output_path = os.path.join(OUTPUT_ROOT, f"{NAME}_size{size}_{MODEL_TYPE}_{method}_tissuenet_norm.tif")
        tifffile.imwrite(output_path, segmentation, compression="zlib")
        n_objects = len(np.unique(segmentation[segmentation > 0]))
        print(f"crop {size}: upsampled {predictor.model.image_size / size:.2f}x by SAM2, {n_objects} objects")
        results[size] = (image, segmentation)
    return method, results


def plot_crop_sizes(method, results):
    # Each crop is drawn at its position inside the largest one, so the rows align.
    largest = max(results)
    fig, axes = plt.subplots(len(results), 3, figsize=(19, 6.5 * len(results)), squeeze=False)
    for row, (size, (image, segmentation)) in zip(axes, results.items()):
        display = (image * 255).astype("uint8")
        overlay = display.copy()
        overlay[find_boundaries(segmentation, mode="inner")] = (255, 255, 0)
        n_objects = len(np.unique(segmentation[segmentation > 0]))
        offset = (largest - size) // 2
        extent = (offset, offset + size, offset + size, offset)
        panels = [
            (display, f"crop {size}x{size}: channel 1, channel 4, zeros"),
            (random_label_colors(int(segmentation.max()))[segmentation], f"{n_objects} objects"),
            (overlay, "Boundaries on the model input"),
        ]
        for ax, (panel, title) in zip(row, panels):
            ax.imshow(panel, extent=extent)
            ax.set_xlim(0, largest)
            ax.set_ylim(largest, 0)
            ax.set_title(title)
    for ax in axes.flat:
        ax.axis("off")
    label = method.upper().replace("_", " ")
    fig.suptitle(f"{NAME} ({MODEL_TYPE}): {label} per crop size", y=1.005)
    fig.tight_layout()
    save_figure(fig, f"{NAME}_{MODEL_TYPE}_{method}_crop_sizes")


def load_region(size):
    image = tifffile.memmap(IMAGE_PATH)  # (C, Y, X)
    if size < min(image.shape[1:]):
        y, x = (slice(center - size // 2, center - size // 2 + size) for center in CENTER)
        image = image[:, y, x]
    membrane, nucleus = np.asarray(image[0]), np.asarray(image[3])
    # Raw channels, the library normalizes every tile by its own 2-98 percentiles.
    return np.stack([membrane, nucleus, np.zeros_like(membrane)], axis=-1)


def large_region_path(size, density_threshold, global_norm):
    method = "ais_sparse" if density_threshold is None else f"ais_sparse_density_threshold_{density_threshold:g}"
    suffix = "_global_norm" if global_norm else ""
    return os.path.join(OUTPUT_ROOT, f"Restain_Day7_2_S190131_region{size}_{MODEL_TYPE}_{method}{suffix}.tif")


def normalize_globally(raw):
    image = np.zeros(raw.shape, dtype="float32")
    for c, name in enumerate(("channel 1", "channel 4")):
        bounds = compute_percentile_bounds(raw[..., c])
        image[..., c] = normalize_raw(raw[..., c], bounds=bounds)
        print(f"Global bounds of {name}: p2 {bounds[0].item():.1f}, p98 {bounds[1].item():.1f}")
    return image


@contextmanager
def without_tile_normalization():
    # The tiled 2d embeddings normalize every tile by its own 2-98 percentiles. This keeps the input's normalization.
    tile_shapes = []
    per_tile_normalization = batched_inference.to_image

    def to_uint8(tile):
        tile_shapes.append(tile.shape)
        return np.round(np.clip(tile, 0, 1) * 255).astype("uint8")

    batched_inference.to_image = to_uint8
    try:
        yield tile_shapes
    finally:
        batched_inference.to_image = per_tile_normalization


def run_large_region(size, density_threshold, global_norm):
    raw = load_region(size)
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode="ais", is_tiled=True)
    postprocessing = {} if density_threshold is None else {"density_threshold": density_threshold}
    start = time.time()
    if global_norm:
        image = normalize_globally(raw)
        with without_tile_normalization() as tile_shapes:
            segmentation = automatic_instance_segmentation(
                predictor, segmenter, input_path=image, ndim=2, tile_shape=LARGE_TILE_SHAPE, halo=LARGE_HALO,
                batch_size=None, **postprocessing,
            )
        print(f"{len(tile_shapes)} tiles were encoded with the global normalization")
    else:
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=raw, ndim=2, tile_shape=LARGE_TILE_SHAPE, halo=LARGE_HALO,
            batch_size=None, **postprocessing,
        )
    duration = time.time() - start
    output_path = large_region_path(raw.shape[0], density_threshold, global_norm)
    tifffile.imwrite(output_path, segmentation, compression="zlib")
    n_objects = len(np.unique(segmentation[segmentation > 0]))
    print(f"region {raw.shape[:2]}: {n_objects} objects in {duration:.1f} s, saved to {output_path}")


def plot_large_region(size, density_threshold, global_norm):
    raw = load_region(size)
    output_path = large_region_path(raw.shape[0], density_threshold, global_norm)
    segmentation = tifffile.imread(output_path)

    inside = segmentation > 0
    areas = np.bincount(segmentation[inside])
    ids = np.flatnonzero(areas)
    mean_membrane = np.bincount(segmentation[inside], weights=raw[..., 0][inside])[ids] / areas[ids]
    dark_ids = ids[mean_membrane < DARK_THRESHOLD]
    print(f"{len(ids)} objects, {len(dark_ids)} on dark background (mean raw channel 1 < {DARK_THRESHOLD})")

    display = to_image(raw)
    boundaries = find_boundaries(segmentation, mode="thick")
    overlay = display.copy()
    overlay[boundaries] = (255, 255, 0)
    overlay[boundaries & np.isin(segmentation, dark_ids)] = (255, 0, 255)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12.5))
    axes[0].imshow(overlay)
    axes[0].set_title(f"{len(ids)} objects, {len(dark_ids)} on dark background (magenta)")
    axes[1].imshow(random_label_colors(int(segmentation.max()))[segmentation])
    axes[1].set_title("Instances")
    for ax in axes:
        ax.axis("off")
    title = os.path.splitext(os.path.basename(output_path))[0]
    normalization = "global" if global_norm else "per tile"
    fig.suptitle(
        f"{title}: tiled {LARGE_TILE_SHAPE} + halo {LARGE_HALO}, {normalization} 2-98 normalization, "
        f"centered at (y, x) = {CENTER}", y=1.0
    )
    fig.tight_layout()
    plot_path = os.path.join(PLOT_ROOT, f"{title}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved the plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", nargs="+", choices=list(INPUT_CHANNELS), default=["tissuenet_norm"])
    parser.add_argument("--preview", action="store_true", help="Plot the crop and the model inputs.")
    parser.add_argument("-e", "--engines", nargs="+", choices=["apg", "ais"], default=[], help="The engines to run.")
    parser.add_argument("--large_regions", nargs="+", type=int, help="Run tiled AIS sparse on these region sizes.")
    parser.add_argument("--global_norm", action="store_true", help="Normalize large regions globally, not per tile.")
    parser.add_argument("--plot_saved", action="store_true", help="Plot the saved large regions instead of running.")
    parser.add_argument("--crop_sizes", nargs="+", type=int, help="Run AIS sparse on centered crops of these sizes.")
    parser.add_argument("--density_threshold", type=float, help="The AIS seed threshold, by default the library's.")
    parser.add_argument("-m", "--modes", nargs="+", choices=["sparse", "dense"], default=["sparse"], help="AIS modes.")
    parser.add_argument("--check_encoder_input", action="store_true", help="Check what the encoder receives.")
    parser.add_argument("--predictions", action="store_true", help="Plot the foreground and distances of AIS.")
    parser.add_argument("--apg_normalizations", action="store_true", help="Run APG for every input normalization.")
    parser.add_argument("--sweep", action="store_true", help="Sweep the AIS sparse post-processing parameters.")
    parser.add_argument("--scales", action="store_true", help="Run APG and AIS at smaller object scales.")
    parser.add_argument("--channel_gains", action="store_true", help="Run APG and AIS for scaled channels.")
    parser.add_argument("--compare_tissuenet", action="store_true", help="Compare input statistics with TissueNet.")
    parser.add_argument("-n", "--normalizations", nargs="+", choices=list(NORMALIZATIONS), default=["p2_p98"])
    args = parser.parse_args()

    crop, overview = load_data()
    images = {input_name: to_model_input(crop, input_name) for input_name in args.inputs}
    for input_name, image in images.items():
        print_intensities(crop, image, input_name)
    if args.preview:
        plot_crop(crop, overview, images)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for engine in args.engines:
        run_segmentation(crop, images, engine, args.modes, args.check_encoder_input, args.density_threshold)
    if args.predictions:
        for input_name in args.inputs:
            results = get_normalization_predictions(crop, input_name, args.normalizations, args.check_encoder_input)
            plot_normalizations(input_name, results)
            plot_watershed(input_name, results)
    if args.compare_tissuenet:
        compare_tissuenet(crop)
    if args.large_regions:
        for size in args.large_regions:
            if args.plot_saved:
                plot_large_region(size, args.density_threshold, args.global_norm)
            else:
                run_large_region(size, args.density_threshold, args.global_norm)
    if args.crop_sizes:
        plot_crop_sizes(*run_crop_sizes(args.crop_sizes, args.density_threshold))
    if args.scales:
        plot_scales(run_scales())
    if args.channel_gains:
        plot_channel_gains(*run_channel_gains(crop))
    if args.sweep:
        for input_name in args.inputs:
            plot_sweep(input_name, *sweep_postprocessing(crop, input_name))
    if args.apg_normalizations:
        for input_name in args.inputs:
            results = run_apg_normalizations(crop, input_name, args.normalizations, args.check_encoder_input)
            plot_apg_normalizations(input_name, results)


if __name__ == "__main__":
    main()
