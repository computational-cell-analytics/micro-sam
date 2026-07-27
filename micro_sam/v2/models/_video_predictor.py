import os
import contextlib
from collections import OrderedDict
from typing import Optional, Dict, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import _load_checkpoint
from sam2.utils.misc import AsyncVideoFrameLoader
from sam2.sam2_video_predictor import SAM2VideoPredictor

import torch


# Number of recent frames whose precomputed features stay cached on the device during inference. >1
# so segmenting the same or nearby slice again reuses the upload. Small so memory stays bounded.
MAX_CACHED_FRAMES = 8

# The ImageNet statistics SAM2 normalizes its frames with (sam2.utils.misc only has them as defaults).
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def _load_img_as_tensor(img_path, image_size):
    """Load a single frame as a float32 [0, 1] tensor of shape (3, image_size, image_size).

    File-path and numpy inputs are both percentile-normalized per channel, so that any input dtype
    is mapped to the range SAM2's ImageNet normalization expects. Both also preserve aspect ratio:
    the longest side is resized to ``image_size`` and the remaining bottom/right region is zero-padded.

    Returns:
        img: (3, image_size, image_size) float32 tensor, ImageNet-normalised by the caller.
        video_height: max(H, W) of the original frame - used as the effective square dimension
            for SAM2's coordinate normalization so prompts map into the resized content region.
        video_width: same as video_height.
    """
    from micro_sam.v2.normalization import normalize_raw

    if isinstance(img_path, str):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB"))
    else:
        img_np = img_path
        img_np = np.stack([img_np] * 3, axis=-1) if img_np.ndim == 2 else img_np

    img_np = normalize_raw(img_np, axis=(0, 1))

    # The effective square size gives prompts one isotropic scale factor.
    from micro_sam.v2.transforms.resize import resize_longest_side_and_pad_tensor
    H, W = img_np.shape[:2]
    video_height = video_width = max(H, W)
    img = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
    img, _ = resize_longest_side_and_pad_tensor(img[None], image_size)
    img = img[0]
    return img, video_height, video_width


def _prepare_frame(raw, image_size):
    """Resize and ImageNet-normalize one frame, exactly as the video predictor loads its frames.

    Args:
        raw: The frame, either a numpy array or a path to an image file.
        image_size: The size the longest side is resized to.

    Returns:
        The frame as a (3, image_size, image_size) float32 tensor on the CPU.
    """
    image, _, _ = _load_img_as_tensor(raw, image_size)
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    return (image - mean) / std


class _LazyVideoFrames:
    """Produce per-slice frame tensors from a volume on demand, without stacking the whole volume.

    'inference_state["images"]' is only ever integer-indexed and passed to 'len', so a sequence that
    resizes + normalises one slice at access time is a drop-in for the eager
    '(num_frames, 3, image_size, image_size)' tensor. This avoids holding every slice at 'image_size^2'
    (~12 MB/slice), which otherwise grows unbounded with depth and, after the embeddings were moved to
    disk, is the dominant per-volume RAM cost. Frames are returned on CPU (the consumer moves the
    current frame to the device); the upstream <=MAX_CACHED_FRAMES feature cache already retains the
    frames in active use, so nothing is cached here. A lazy volume (dask / zarr / h5py) is kept as it
    was handed over and read one slice at a time, so it never has to fit in host RAM.
    """

    def __init__(self, volume, image_size, img_mean, img_std):
        self._volume = volume
        self._image_size = image_size
        self._img_mean = img_mean
        self._img_std = img_std
        h, w = volume.shape[1], volume.shape[2]
        self.video_height = self.video_width = max(h, w)

    def __len__(self):
        return int(self._volume.shape[0])

    def __getitem__(self, index):
        # Read and convert only the requested slice, so a lazy volume stays lazy.
        img, _, _ = _load_img_as_tensor(np.asarray(self._volume[index]), self._image_size)
        return (img - self._img_mean) / self._img_std


def _load_video_frames_from_images(
    video_path,
    volume,
    image_size,
    offload_video_to_cpu,
    img_mean=IMG_MEAN,
    img_std=IMG_STD,
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    verbosity=True,
):
    """Based on 'load_video_frames_from_jpg_images'.

    Returns the frame sequence (resized to image_size x image_size and ImageNet-normalised) plus the
    effective video height / width, for two input kinds:

    - `volume` (a (Z, Y, X) array-like, the micro-sam path): returns a lazy `_LazyVideoFrames` that
      reads, resizes and normalises one slice on demand on CPU, so the whole volume is never read or
      stacked at image_size^2 in memory. A lazy input (dask / zarr / h5py) therefore stays lazy.
      `offload_video_to_cpu` does not apply here (the consumer moves the current frame to the device
      per access).
    - `video_path` (a directory of "<frame_index>.jpg" files): stacks the frames into a single tensor,
      on the GPU if `offload_video_to_cpu` is `False` else on CPU. Set `async_loading_frames` to `True`
      to load these frames asynchronously.
    """
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if video_path is None:
        # Read the shape instead of the data, so a lazy input (dask / zarr / h5py) stays lazy.
        shape = tuple(volume.shape)
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D volume of shape (Z, Y, X), got an array of shape {shape}.")
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

    def _wait_for_offloaded_state(self, inference_state):
        """Wait for the offloaded tensors to reach the CPU before anything reads them back.

        With 'offload_state_to_cpu' SAM2 stores 'pred_masks' and 'maskmem_features' on the CPU with a
        'non_blocking' copy into pageable memory, which records no event to wait on. Without a wait a
        host reader can observe the buffer the allocator recycled from the previous call.

        This is a host barrier, so it is called only where a host read follows: the consolidation
        across objects and the dtype restore below. Propagation hands out the on-device masks and
        copies the offloaded tensors back on the stream that wrote them, so it needs no wait per object
        and frame. The wait covers that stream rather than the whole device, so a second model replica
        in the batched pipeline keeps running.
        """
        if not inference_state.get("offload_state_to_cpu"):
            return
        device = torch.device(inference_state["device"])
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def _autocasts(self, inference_state) -> bool:
        """Whether this state runs the model in SAM2's trained bfloat16 precision.

        This selects the precision, never the device: hardware excluded here still runs where it ran
        before, just in fp32. It has to agree with what torch actually does, because '_run_*' below
        skips the mask-memory restore whenever this is True - claiming an autocast that torch then
        declines would put bfloat16 memory back in front of fp32 projections.

        CUDA needs *native* bfloat16, which is Ampere and newer (compute capability 8.0): older GPUs
        emulate it more slowly than the fp32 they run today, and torch raises where even the emulation
        is missing. MPS has it from macOS 14, which is exactly the version torch gates on - below that
        it warns and silently disables the autocast. The CPU gains nothing from it and keeps fp32.
        """
        device = torch.device(inference_state["device"])
        if device.type == "cuda":
            return torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8
        if device.type == "mps":
            return torch.backends.mps.is_available() and torch.backends.mps.is_macos_or_newer(14, 0)
        return False

    def _autocast(self, inference_state):
        """Run the model in the precision it was trained in.

        SAM2 is trained and officially run under bfloat16 autocast, which is also what makes its
        bfloat16 mask memory self-consistent. Measured on an A100 MIG partition with 'hvit_t_cells' over
        a 30 slice volume, it propagates 2.9x faster for one object and 3.7x for four, at a foreground
        dice of 0.97 / 0.98 against fp32. Only the video predictor is autocast: the image embeddings are
        precomputed and cached separately, so their stored values - and the cache signature - are
        unaffected by this choice.
        """
        if not self._autocasts(inference_state):
            return contextlib.nullcontext()
        device_type = torch.device(inference_state["device"]).type
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    def _restore_memory_dtype(self, maskmem_features):
        """Undo SAM2's bfloat16 downcast of the mask memory, for the paths that run without autocast.

        SAM2 stores 'maskmem_features' as bfloat16 to shrink the state, which its own inference makes
        safe by running under a bfloat16 autocast. Without one the cast only survives because
        'sam2_base._prepare_memory_conditioned_features' concatenates the fp32 object pointers onto the
        memory and 'torch.cat' promotes the result back to fp32. An object whose only conditioning frame
        lies ahead of the frame being propagated contributes no object pointers, so its memory stays
        bfloat16 and hits the fp32 memory-attention projections. That is reachable from the GUI: batched
        volume segmentation on the CPU with objects prompted on different slices.

        Only call this once the offloaded copy has landed - it reads the tensor on the host.
        """
        if maskmem_features is None:
            return None
        return maskmem_features.to(next(self.parameters()).dtype)

    def _run_memory_encoder(self, inference_state, *args, **kwargs):
        with self._autocast(inference_state):
            maskmem_features, maskmem_pos_enc = super()._run_memory_encoder(inference_state, *args, **kwargs)
        if not self._autocasts(inference_state):
            self._wait_for_offloaded_state(inference_state)
            maskmem_features = self._restore_memory_dtype(maskmem_features)
        return maskmem_features, maskmem_pos_enc

    def _run_single_frame_inference(self, inference_state, *args, **kwargs):
        with self._autocast(inference_state):
            out = super()._run_single_frame_inference(inference_state, *args, **kwargs)
        if not self._autocasts(inference_state):
            self._wait_for_offloaded_state(inference_state)
            out[0]["maskmem_features"] = self._restore_memory_dtype(out[0]["maskmem_features"])
        return out

    def _consolidate_temp_output_across_obj(self, inference_state, *args, **kwargs):
        """Wait once here: this copies every object's offloaded 'pred_masks' into one host buffer.

        It runs when a prompt is added, not inside the propagation loop, so it costs one wait per
        interaction.
        """
        self._wait_for_offloaded_state(inference_state)
        return super()._consolidate_temp_output_across_obj(inference_state, *args, **kwargs)

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
        # (see '_get_image_feature'). Loading every slice's high-resolution features up-front costs
        # about 200 MB per slice and runs out of memory for large volumes. The lazy single-frame cache
        # keeps memory bounded. When the embeddings are backed by a zarr on disk (lazy_loading=True), only one
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
