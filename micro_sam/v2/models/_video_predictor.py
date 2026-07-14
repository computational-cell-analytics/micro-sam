import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Optional, Dict, Union

import numpy as np
from PIL import Image
import torch

from sam2.build_sam import _load_checkpoint
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import AsyncVideoFrameLoader


# Number of recent frames whose precomputed features are cached on the device during inference. >1
# so repeatedly segmenting the same / nearby slice reuses the upload; small so memory stays bounded.
MAX_CACHED_FRAMES = 8


def _load_img_as_tensor(img_path, image_size):
    """Load a single frame as a float32 [0, 1] tensor of shape (3, image_size, image_size).

    File-path and numpy inputs both preserve aspect ratio: the longest side is resized to
    ``image_size`` and the remaining bottom/right region is zero-padded.

    Returns:
        img: (3, image_size, image_size) float32 tensor, ImageNet-normalised by the caller.
        video_height: max(H, W) of the original frame - used as the effective square dimension
            for SAM2's coordinate normalization so prompts map into the resized content region.
        video_width: same as video_height.
    """
    if isinstance(img_path, str):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    else:
        img_np = img_path
        img_np = np.stack([img_np] * 3, axis=-1) if img_np.ndim == 2 else img_np

        # Percentile-normalize each channel to [0, 1], so any input dtype (e.g. uint16 microscopy data)
        # is mapped to the range SAM2's ImageNet normalization expects. Clip since percentile
        # normalization maps the 2nd / 98th percentiles to 0 / 1 and overshoots outside that range.
        from torch_em.transform.raw import normalize_percentile
        img_np = normalize_percentile(img_np.astype(np.float32), lower=2.0, upper=98.0, axis=(0, 1))
        img_np = np.clip(img_np, 0.0, 1.0)

    # The effective square size gives prompts one isotropic scale factor.
    from micro_sam.v2.transforms.resize import resize_longest_side_and_pad_numpy
    H, W = img_np.shape[:2]
    video_height = video_width = max(H, W)
    img_np, _ = resize_longest_side_and_pad_numpy(img_np, image_size)

    img = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
    return img, video_height, video_width


class _LazyVideoFrames:
    """Produce per-slice frame tensors from a numpy volume on demand, without stacking the whole volume.

    'inference_state["images"]' is only ever integer-indexed and passed to 'len', so a sequence that
    resizes + normalises one slice at access time is a drop-in for the eager
    '(num_frames, 3, image_size, image_size)' tensor. This avoids holding every slice at 'image_size^2'
    (~12 MB/slice), which otherwise grows unbounded with depth and, after the embeddings were moved to
    disk, is the dominant per-volume RAM cost. Frames are returned on CPU (the consumer moves the
    current frame to the device); the upstream <=MAX_CACHED_FRAMES feature cache already retains the
    frames in active use, so nothing is cached here.
    """

    def __init__(self, volume, image_size, img_mean, img_std):
        self._volume = volume
        self._image_size = image_size
        self._img_mean = img_mean
        self._img_std = img_std
        h, w = volume.shape[1], volume.shape[2]
        self.video_height = self.video_width = max(h, w)

    def __len__(self):
        return len(self._volume)

    def __getitem__(self, index):
        img, _, _ = _load_img_as_tensor(self._volume[index], self._image_size)
        return (img - self._img_mean) / self._img_std


def _load_video_frames_from_images(
    video_path,
    volume,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    verbosity=True,
):
    """Based on 'load_video_frames_from_jpg_images'.

    Load the video frames from a directory of image files (eg. "<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if video_path is None:
        # Coerce lazy inputs (e.g. dask / zarr / h5py arrays handed over by a napari layer) to a numpy
        # array (cheap at native resolution; the expensive image_size^2 copy is what we avoid stacking).
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume of shape (Z, Y, X), got an array of shape {volume.shape}.")
        # Stream slices lazily (resize + normalise on access) instead of stacking the whole volume at
        # image_size^2, so RAM stays bounded regardless of depth.
        lazy_images = _LazyVideoFrames(volume, image_size, img_mean, img_std)
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    else:
        if isinstance(video_path, str) and os.path.isdir(video_path):
            frames_folder = video_path
        else:
            raise AssertionError("The video predictor expects the user to provide the folder where frames are stored.")

        frame_names = [p for p in os.listdir(frames_folder)]  # NOTE: This part has changed to support multiple ffs.
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"No images found in '{frames_folder}'.")

        img_paths = [os.path.join(frames_folder, frame_name) for frame_name in frame_names]

        if async_loading_frames:
            lazy_images = AsyncVideoFrameLoader(
                img_paths,
                image_size,
                offload_video_to_cpu,
                img_mean,
                img_std,
                compute_device,
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading", disable=not verbosity)):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)

    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    # Normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


class CustomVideoPredictor(SAM2VideoPredictor):
    """The video predictor class inherited from the original predictor class to update 'init_state'.

    Overrides init_state to accept a numpy volume and a precomputed embeddings dict directly,
    bypassing SAM2's default frame-loading path. All other predictor behaviour (add_new_points_or_box,
    propagate_in_video, reset_state, etc.) is inherited unchanged from SAM2VideoPredictor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable responsive interactive correction: a click on an already-tracked frame turns that
        # frame into a conditioning frame (so the correction sticks and propagates), and stale
        # non-conditioning memory around it is cleared. Without these, iterative 3D prompts leave the
        # result unchanged. 'clear_non_cond_mem_around_input' needs the per-object helper we add below.
        self.add_all_frames_to_correct_as_cond = True
        self.clear_non_cond_mem_around_input = True

    @torch.inference_mode()
    def init_state(
        self,
        volume: np.ndarray,
        volume_embeddings: Dict,
        device: Optional[Union[str, torch.device]] = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        verbosity: bool = True,
        ignore_caching_features: bool = False,
    ):
        """Initialize an inference state.

        Args:
            volume: The volume as numpy array in memory.
            volume_embeddings: The precomputed embeddings.
            device: The torch device.
            offload_video_to_cpu: Move the video from GPU to CPU.
            offload_state_to_cpu: Move the inference state components from GPU to CPU.
            async_loading_frames: Asynchronises the frame loading process.
            verbosity: The verbosity argument.
            ignore_caching_features: Avoids ensuring feature caching over all frames.
        """

        from micro_sam.v2.util import _get_device

        # Get the expected device.
        device = _get_device(device)

        # Convert the volume or video in expected format.
        images, video_height, video_width = _load_video_frames_from_images(
            video_path=None,  # NOTE: This feature works. We just don't care about it in our tasks.
            volume=volume,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=device,
            verbosity=verbosity,
        )

        # 'inference_state' is the running dictionary which keeps all key details in memory.
        inference_state = {
            # Initialize the image and frame details.
            "images": images,
            "num_frames": len(images),

            # Whether to offload the video frames to CPU memory.
            # Turning on this option saves the GPU memory with only a very small overhead.
            "offload_video_to_cpu": offload_video_to_cpu,

            # Whether to offload the inference state to CPU memory.
            # Turning on this option saves the GPU memory at the cost of a lower tracking fps
            # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
            # and from 24 to 21 when tracking two objects)
            "offload_state_to_cpu": offload_state_to_cpu,

            # The original video height and width, used for resizing final output scores
            "video_height": video_height,
            "video_width": video_width,
            "device": device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else device,

            # Inputs on each frame
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},

            # Values that don't change across frames (so we only need to hold one copy of them)
            "constants": {},

            # The mapping between client-side object id and model-side object index
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],

            # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
            "output_dict_per_obj": {},

            # A temporary storage to hold new outputs when user interact with a frame
            # to add clicks or mask (it's merged into "output_dict" before propagation starts)
            "temp_output_dict_per_obj": {},

            # Frames that already holds consolidated outputs from click or mask inputs
            # (we directly use their consolidated outputs during tracking)
            # metadata for each tracking frame (e.g. which direction it's tracked)
            "frames_tracked_per_obj": {},
        }

        # Avoids preparing cached features - essential for the embedding precomputation stage.
        if ignore_caching_features:
            inference_state["cached_features"] = {}  # Create an empty 'cached_features' dictionary to warm up.
            return inference_state

        # Store the precomputed embeddings and load each frame's features lazily during tracking
        # (see '_get_image_feature'). Materialising every slice's high-resolution features up-front
        # costs ~200 MB/slice and OOMs for large volumes; the lazy single-frame cache keeps memory
        # bounded. When the embeddings are backed by a zarr on disk (lazy_loading=True), only one
        # slice is held in memory at a time.
        inference_state["precomputed_embeddings"] = volume_embeddings
        inference_state["cached_features"] = {}

        return inference_state

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute or look up the image features for a frame.

        Overrides 'SAM2VideoPredictor._get_image_feature' to source per-frame features from the
        precomputed embeddings (if stored on the inference state) instead of running the image
        encoder. A single-frame cache bounds memory for large volumes. Falls back to the parent
        behaviour (run the encoder) when no precomputed embeddings are available.
        """
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            embeddings = inference_state.get("precomputed_embeddings")
            if embeddings is None:
                return super()._get_image_feature(inference_state, frame_idx, batch_size)

            from micro_sam.v2.util import _to_device_tensor
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            # In-memory embeddings keep 'pos_enc'/'fpn' as device tensors, which 'np.asarray' cannot
            # convert (fails on mps/cuda); '_to_device_tensor' handles both tensors and numpy/zarr.
            vision_pos_enc = [_to_device_tensor(t[frame_idx], device) for t in embeddings["pos_enc"]]
            backbone_fpn = [_to_device_tensor(t[frame_idx], device) for t in embeddings["fpn"]]
            backbone_out = {"backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc}
            # Cache the few most recent frames (not just one) so repeatedly segmenting the same or a
            # nearby slice does not re-read the (possibly zarr-backed, tiled) embeddings and re-upload
            # them to the device every interaction - the cause of the noticeable per-slice delay with
            # tiling. The small cap still bounds memory for large volumes / propagation.
            cache = inference_state["cached_features"]
            cache[frame_idx] = (image, backbone_out)
            while len(cache) > MAX_CACHED_FRAMES:
                del cache[next(iter(cache))]  # evict the oldest inserted frame (FIFO)

        # Expand the features to the number of objects being tracked (mirrors upstream SAM2).
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(batch_size, -1, -1, -1)

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _clear_obj_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
        """Clear one object's non-conditioning memory around an interacted frame.

        The installed SAM2 fork calls this per-object variant from 'propagate_in_video' and
        'propagate_in_video_preflight' (guarded by 'clear_non_cond_mem_around_input') but only ships
        the global '_clear_non_cond_mem_around_input', so enabling the flag raises AttributeError. We
        restore the per-object method here rather than editing the fork. Dropping the stale surrounding
        non-conditioning memory lets correction clicks actually take effect during iterative prompting.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        non_cond_frame_outputs = inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)


def _build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    hydra_overrides = [
        "++model._target_=micro_sam.v2.models._video_predictor.CustomVideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks
            # are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area`
            # (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model
