"""FOCUS-3D as an automatic 3d segmentation baseline.

FOCUS-3D (Zhang et al., bioRxiv 2026) segments 3d fluorescence volumes with a Mask2Former head on a
MAE ViT-Adapter backbone. Segmentation is left entirely to its own `infer_volume`, so this module is
only the adapter the evaluation needs: it serves the volume from memory and caches the model that
`infer_volume` would otherwise rebuild for every volume.

So that the method is scored the way its authors run it, the patch size comes from the checkpoint
config and the parameters default to the values the napari plugin ships, not to the stale fallbacks
in the `infer_volume` signature. Everything not listed here keeps FOCUS-3D's own default.
"""

import os
import tempfile
from contextlib import contextmanager

import numpy as np

# Despite the file name, `inference_win` is the cross-platform pure-PyTorch backend, so it needs
# neither detectron2 nor the compiled MSDeformAttn kernel.
from focus3d.segmentation.FOCUS3D import inference_win

FOCUS3D_CHECKPOINT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/focus3d"
FOCUS3D_CONFIG = "configs/3d_test.yaml"

# The three released checkpoints, see https://huggingface.co/Qinghua-thu/FOCUS-3D.
FOCUS3D_CHECKPOINTS = {
    "general": "model_final.pth",
    "membrane": "model_final_membrane.pth",
    "nuclei": "model_final_nuclei.pth",
}

# The radius the model resamples every volume to; the reference that `cell_radius` is relative to.
REFERENCE_CELL_RADIUS = 15.0

# The napari plugin defaults, which differ from the `infer_volume` signature for the stride, the
# score threshold and the minimum area. The patch size is absent: it is read from the config.
FOCUS3D_DEFAULTS = {
    "z_ratio": 1.0,
    "cell_radius": REFERENCE_CELL_RADIUS,
    "background_threshold": 1.0,
    "lower_percentile": 1.0,
    "upper_percentile": 99.0,
    "stride": (24, 64, 64),
    # Throughput only, though not bit-reproducible: the batch shape sets the fp16 reduction order.
    "batch_size": 16,
    "data_loader_num_workers": 4,
    "score_thresh": 0.7,
    "mask_thresh": 0.5,
    "min_edge_area": 64,
    "size_filter_min_size": 0,
    "size_filter_max_size": 100000,
}


def resolve_checkpoint(model_type):
    """Resolve a FOCUS-3D model name to its checkpoint path."""
    if model_type not in FOCUS3D_CHECKPOINTS:
        raise ValueError(f"Unknown FOCUS-3D model '{model_type}'; expected one of {sorted(FOCUS3D_CHECKPOINTS)}.")
    return os.path.join(FOCUS3D_CHECKPOINT_ROOT, FOCUS3D_CHECKPOINTS[model_type])


def cache_built_models():
    """Make `infer_volume` reuse one model per checkpoint instead of rebuilding it for every volume.

    The backbone is a 24-layer ViT behind a 4.5 GB checkpoint, so rebuilding it per sample would
    dominate the runtime of an evaluation. Patching the one function `infer_volume` builds through
    leaves the rest of its pipeline untouched.
    """
    if getattr(inference_win, "build_predictor_is_cached", False):
        return

    build_predictor = inference_win.build_predictor
    cache = {}

    def cached_build_predictor(cfg):
        key = (str(cfg.MODEL.WEIGHTS), str(cfg.MODEL.DEVICE))
        if key not in cache:
            cache[key] = build_predictor(cfg)
        return cache[key]

    inference_win.build_predictor = cached_build_predictor
    inference_win.build_predictor_is_cached = True


@contextmanager
def volume_from_memory(volume):
    """Serve `infer_volume` its volume from memory rather than from a file.

    This also avoids a bug in FOCUS-3D: `read_volume` returns a read-only memmap, which
    `normalize_and_pad_volume` clips in place once the volume is float32 and goes unresampled.
    """
    read_volume = inference_win.read_volume
    inference_win.read_volume = lambda image_path: volume
    try:
        yield
    finally:
        inference_win.read_volume = read_volume


class Focus3DSegmenter:
    """FOCUS-3D automatic 3d instance segmentation over in-memory volumes.

    Args:
        model_type: Which released checkpoint to use, 'general', 'membrane' or 'nuclei'.
        checkpoint: An explicit checkpoint path, overriding `model_type`.
        config_file: The FOCUS-3D config. Defaults to the packaged `configs/3d_test.yaml`.
        device: The torch device. Defaults to FOCUS-3D's own resolution, i.e. cuda:0 or cpu.
        **overrides: Inference parameters overriding `FOCUS3D_DEFAULTS`.
    """

    def __init__(self, model_type="general", checkpoint=None, config_file=None, device=None, **overrides):
        cache_built_models()

        unknown = set(overrides) - set(FOCUS3D_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FOCUS-3D parameters: {sorted(unknown)}.")
        self.params = {**FOCUS3D_DEFAULTS, **overrides}
        self.params["stride"] = tuple(self.params["stride"])

        if checkpoint is None:
            checkpoint = resolve_checkpoint(model_type)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"The FOCUS-3D checkpoint '{checkpoint}' does not exist. The weights are gated; "
                "accept the terms at https://huggingface.co/Qinghua-thu/FOCUS-3D and download them."
            )
        self.checkpoint = checkpoint
        if config_file is None:
            config_file = os.path.join(os.path.dirname(inference_win.__file__), FOCUS3D_CONFIG)
        self.config_file = config_file

        self.device = inference_win.activate_inference_device(device)
        cfg = inference_win.setup_cfg(self.config_file, self.checkpoint, device=self.device)
        # The window the backbone was trained on. FOCUS-3D exposes only the stride around it.
        self.patch_size = tuple(int(size) for size in cfg.MODEL.BACKBONE.IMG_SIZE)
        inference_win.build_predictor(cfg)  # Load the weights now, not on the first volume.

    def __call__(self, volume, z_ratio=None, cell_radius=None):
        """Segment one volume.

        Args:
            volume: The 3d volume in (Z, Y, X).
            z_ratio: The physical z-to-xy spacing ratio. Defaults to the plugin's 1.0.
            cell_radius: The cell radius in the xy plane, in pixels of `volume`. The volume is
                resampled so that it becomes REFERENCE_CELL_RADIUS. Defaults to the plugin's 15.0.

        Returns:
            The instance segmentation, in the shape of `volume`.
        """
        volume = np.asarray(volume, dtype="float32")
        if volume.ndim != 3:
            raise ValueError(f"FOCUS-3D expects a 3d volume in (Z, Y, X), got shape {volume.shape}.")
        if not volume.flags.writeable:  # FOCUS-3D normalizes the volume in place.
            volume = volume.copy()

        params = dict(self.params)
        if z_ratio is not None:
            params["z_ratio"] = float(z_ratio)
        if cell_radius is not None:
            params["cell_radius"] = float(cell_radius)

        # `infer_volume` writes the instance map next to its input, so it gets a scratch directory.
        # The result is taken from its return value rather than from that file.
        with tempfile.TemporaryDirectory() as tmp_dir, volume_from_memory(volume):
            result = inference_win.infer_volume(
                image_path=os.path.join(tmp_dir, "volume.tif"), config_file=self.config_file,
                weights_path=self.checkpoint, output_dir=tmp_dir, device=self.device,
                patch_size=self.patch_size, **params,
            )
        return result["instance_map"].astype("uint32")
