import contextlib
from collections import OrderedDict
from collections.abc import Mapping
from typing import Optional, Dict, Union

import numpy as np
from tqdm import tqdm

from sam2.build_sam import _load_checkpoint
from sam2.sam2_video_predictor import SAM2VideoPredictor

import torch


# Number of recent frames whose precomputed features stay cached on the device during inference. >1
# so segmenting the same or nearby slice again reuses the upload. Small so memory stays bounded.
MAX_CACHED_FRAMES = 8

# The ImageNet statistics SAM2 normalizes its frames with (sam2.utils.misc only has them as defaults).
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def _load_frame_as_tensor(raw, image_size):
    """Load a single frame as a float32 [0, 1] tensor of shape (3, image_size, image_size).

    The frame is percentile-normalized per channel, so that any input dtype is mapped to the range
    SAM2's ImageNet normalization expects, and it keeps its aspect ratio: the longest side is resized
    to `image_size` and the remaining bottom/right region is zero-padded. The caller applies the
    ImageNet normalization.
    """
    from micro_sam.v2.normalization import normalize_raw
    from micro_sam.v2.transforms.resize import resize_longest_side_and_pad_tensor

    img_np = np.stack([raw] * 3, axis=-1) if raw.ndim == 2 else raw
    img_np = normalize_raw(img_np, axis=(0, 1))
    img = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
    img, _ = resize_longest_side_and_pad_tensor(img[None], image_size)
    return img[0]


def _prepare_frame(raw, image_size):
    """Resize and ImageNet-normalize one frame, exactly as the video predictor loads its frames.

    Args:
        raw: The frame as a numpy array.
        image_size: The size the longest side is resized to.

    Returns:
        The frame as a (3, image_size, image_size) float32 tensor on the CPU.
    """
    image = _load_frame_as_tensor(raw, image_size)
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    return (image - mean) / std


def _volume_geometry(volume):
    """The frame count and the effective square size of a (Z, Y, X) volume.

    That is all the inference state needs from its frames: the per-frame features come from the
    precomputed embeddings, never from the volume itself. Only the shape is read, so a lazy input
    (dask / zarr / h5py) is never materialized. The square size gives prompts one isotropic scale
    factor, matching how a frame would be resized and padded.
    """
    shape = tuple(volume.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D volume of shape (Z, Y, X), got an array of shape {shape}.")
    num_frames, height, width = (int(s) for s in shape)
    return num_frames, max(height, width)


BATCHED_FRAME_OUTPUT_KEYS = ("maskmem_features", "pred_masks", "obj_ptr", "object_score_logits")


def _batch_frame_outputs(entries):
    """Concatenate one frame's per-object memory entries along the batch axis."""
    batched = {}
    for key in BATCHED_FRAME_OUTPUT_KEYS:
        values = [entry[key] for entry in entries]
        batched[key] = None if values[0] is None else torch.cat(values, dim=0)
    position = entries[0]["maskmem_pos_enc"]
    # The same for every object, so the batch axis of the group is all it needs, see
    # 'SAM2VideoPredictor._get_maskmem_pos_enc'.
    batched["maskmem_pos_enc"] = (
        None if position is None else [x[0:1].expand(len(entries), -1, -1, -1) for x in position]
    )
    return batched


def _slice_frame_output(batched, index):
    """One object's entry out of a batched frame output, as views rather than copies."""
    entry = {}
    for key in BATCHED_FRAME_OUTPUT_KEYS:
        value = batched[key]
        entry[key] = None if value is None else value[index:index + 1]
    position = batched["maskmem_pos_enc"]
    entry["maskmem_pos_enc"] = None if position is None else [x[0:1] for x in position]
    return entry


class _BatchedMemory(Mapping):
    """The memories of several objects, concatenated on the batch axis as they are read.

    Only the frames the memory selection reaches are concatenated, so the cost follows the memory
    window rather than the length of the volume.
    """

    def __init__(self, per_object):
        self._per_object = per_object
        self._batched = {}

    def __getitem__(self, frame_idx):
        if frame_idx not in self._batched:
            self._batched[frame_idx] = _batch_frame_outputs([entry[frame_idx] for entry in self._per_object])
        return self._batched[frame_idx]

    def __iter__(self):
        return iter(self._per_object[0])

    def __len__(self):
        return len(self._per_object[0])


def _allocated(device):
    """Bytes currently held by tensors on the device, or 0 where that cannot be read."""
    if torch.device(device).type != "cuda":
        return 0
    return torch.cuda.memory_allocated(torch.device(device))


def _cache_capacity(device, entry_bytes, num_frames, share=0.25):
    """How many slices of features to keep, given a share of what is free on the device.

    A propagation pass walks every slice, so a cache shorter than the volume is never hit: what
    matters is whether the whole volume fits, not how close to it one gets. Never fewer than
    MAX_CACHED_FRAMES, so this can only improve on the fixed cap.
    """
    if not entry_bytes or torch.device(device).type != "cuda":
        return MAX_CACHED_FRAMES
    try:
        free, _ = torch.cuda.mem_get_info(torch.device(device))
    except (RuntimeError, AssertionError):
        return MAX_CACHED_FRAMES
    affordable = int(free * share) // entry_bytes
    return int(min(num_frames, max(MAX_CACHED_FRAMES, affordable)))


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

    # Set for the duration of 'skip_prompt_output' only, on the class so that an instance built
    # without '__init__' still reads it.
    _skip_prompt_output = False

    @contextlib.contextmanager
    def skip_prompt_output(self):
        """Skip the preview masks that adding a prompt returns, for callers that discard them.

        Adding a prompt ends by consolidating every object's mask on that frame into one
        video-resolution tensor, only to return it. Consolidating reads the state and writes nothing,
        so leaving it out changes no result at all - it just does not build a tensor nobody asked for.
        The cost is quadratic in the objects of a pass, since every one of them consolidates across
        all the others, which is what makes it worth skipping when prompts are pushed in bulk.
        """
        self._skip_prompt_output = True
        try:
            yield
        finally:
            self._skip_prompt_output = False

    def _wait_for_offloaded_state(self, inference_state):
        """Wait for the offloaded tensors to reach the CPU before anything reads them back.

        With 'offload_state_to_cpu' SAM2 stores 'pred_masks' and 'maskmem_features' on the CPU with a
        'non_blocking' copy into pageable memory, which records no event to wait on. Without a wait a
        host reader can observe the buffer the allocator recycled from the previous call.

        This is a host barrier, so it is called only where a host read follows: the consolidation
        across objects, the dtype restore below and the batched-memory concatenation during propagation.
        The wait covers the current stream rather than the whole device, so a second model replica in
        the batched pipeline keeps running.
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
        is missing. MPS runs bfloat16 from macOS 14, but without tensor cores it gains no throughput
        from it: measured on an M-series Mac over an 8 slice volume it propagates in 2.48s against
        1.84s in fp32, at a foreground IoU of 0.98. The CPU gains nothing from it either.
        """
        device = torch.device(inference_state["device"])
        if device.type == "cuda":
            return torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8
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
        if self._skip_prompt_output:
            return {"pred_masks_video_res": None, "pred_masks": None}
        self._wait_for_offloaded_state(inference_state)
        return super()._consolidate_temp_output_across_obj(inference_state, *args, **kwargs)

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """Pass the skipped consolidation through, so no mask is resized for a discarded output."""
        if any_res_masks is None:
            return None, None
        return super()._get_orig_video_res_output(inference_state, any_res_masks)

    @torch.inference_mode()
    def init_state(
        self,
        volume: np.ndarray,
        volume_embeddings: Dict,
        device: Optional[Union[str, torch.device]] = None,
        offload_state_to_cpu: bool = False,
        max_cached_frames: Optional[int] = None,
    ):
        """Initialize an inference state.

        Args:
            volume: The volume as numpy array in memory.
            volume_embeddings: The precomputed embeddings.
            device: The torch device.
            offload_state_to_cpu: Move the inference state components from GPU to CPU.
            max_cached_frames: How many slices of features stay on the device. A propagation pass
                walks every slice, so a cache shorter than the volume is never hit and every slice is
                read again on every pass. None sizes it to the volume where a quarter of the free
                device memory holds it, and to MAX_CACHED_FRAMES where it does not.
        """

        from micro_sam.v2.util import _get_device

        # Get the expected device.
        device = _get_device(device)

        # The frames themselves are never needed, only their count and their square size.
        num_frames, video_size = _volume_geometry(volume)

        # 'inference_state' is the running dictionary which keeps all key details in memory.
        inference_state = {
            "num_frames": num_frames,

            # Whether to offload the inference state to CPU memory.
            # Turning on this option saves the GPU memory at the cost of a lower tracking fps
            # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
            # and from 24 to 21 when tracking two objects)
            "offload_state_to_cpu": offload_state_to_cpu,

            # The original video height and width, used for resizing final output scores
            "video_height": video_size,
            "video_width": video_size,
            "device": device,
            "max_cached_frames": max_cached_frames,
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
        precomputed embeddings the state always carries, instead of running the image encoder. The
        cache bounds how many slices of them stay on the device, see 'max_cached_frames'.

        The frame itself is returned as None rather than read, resized and uploaded: every caller in
        SAM2 and here discards it ('_, _, vision_feats, ...'), and it is a third of what a cached
        slice would otherwise cost.
        """
        backbone_out = inference_state["cached_features"].get(frame_idx)
        if backbone_out is None:
            embeddings = inference_state["precomputed_embeddings"]

            from micro_sam.v2.util import _to_device_tensor, _shared_pos_enc, _backbone_fpn
            device = inference_state["device"]
            allocated_before = _allocated(device)
            # In-memory embeddings keep 'pos_enc'/'fpn' as device tensors, which 'np.asarray' cannot
            # convert (fails on mps/cuda); '_to_device_tensor' handles both tensors and numpy/zarr.
            vision_pos_enc = [_to_device_tensor(_shared_pos_enc(t), device) for t in embeddings["pos_enc"]]
            vision_features = _to_device_tensor(embeddings["features"][frame_idx], device)
            backbone_fpn = _backbone_fpn(
                [_to_device_tensor(t[frame_idx], device) for t in embeddings["fpn"]], vision_features
            )
            backbone_out = {"backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc}
            allocated_by_entry = _allocated(device) - allocated_before
            # Cache the few most recent frames (not just one) so repeatedly segmenting the same or a
            # nearby slice does not re-read the (possibly zarr-backed, tiled) embeddings and re-upload
            # them to the device every interaction - the cause of the noticeable per-slice delay with
            # tiling. The small cap still bounds memory for large volumes / propagation.
            cache = inference_state["cached_features"]
            cache[frame_idx] = backbone_out
            if inference_state.get("max_cached_frames") is None:
                # Sized from what a slice really costs on the device, measured rather than estimated:
                # the entry holds upcast copies of the stored embeddings, not the stored bytes.
                inference_state["max_cached_frames"] = _cache_capacity(
                    device, allocated_by_entry, inference_state["num_frames"]
                )
            while len(cache) > inference_state["max_cached_frames"]:
                del cache[next(iter(cache))]  # evict the oldest inserted frame (FIFO)

        # Expand the features to the number of objects being tracked (mirrors upstream SAM2).
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(batch_size, -1, -1, -1)

        features = self._prepare_backbone_features(expanded_backbone_out)
        return (None,) + features

    def _memory_groups(self, inference_state, obj_indices):
        """Group the objects that can share a forward pass, i.e. that read the same memory frames.

        The memory selection depends on which frames an object has, never on what is in them, so
        objects with the same frames select the same memory and can go through the model together.
        Objects prompted on different slices fall into groups of their own, which is SAM2's behaviour.
        """
        groups = {}
        for obj_idx in obj_indices:
            output_dict = inference_state["output_dict_per_obj"][obj_idx]
            signature = (
                frozenset(output_dict["cond_frame_outputs"]), frozenset(output_dict["non_cond_frame_outputs"])
            )
            groups.setdefault(signature, []).append(obj_idx)
        return list(groups.values())

    def _track_frame_batch(self, inference_state, obj_indices, frame_idx, reverse):
        """Track one frame for a group of objects in a single forward, storing each object's output."""
        per_object = [inference_state["output_dict_per_obj"][obj_idx] for obj_idx in obj_indices]
        output_dict = {
            "cond_frame_outputs": _BatchedMemory([entry["cond_frame_outputs"] for entry in per_object]),
            "non_cond_frame_outputs": _BatchedMemory([entry["non_cond_frame_outputs"] for entry in per_object]),
        }
        current_out, pred_masks = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=output_dict,
            frame_idx=frame_idx,
            batch_size=len(obj_indices),
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=True,
        )
        for index, output in enumerate(per_object):
            output["non_cond_frame_outputs"][frame_idx] = _slice_frame_output(current_out, index)
        return pred_masks

    @torch.inference_mode()
    def propagate_in_video(
        self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False,
    ):
        """Propagate the prompts through the volume, tracking a frame's objects in one forward pass.

        SAM2 runs one object at a time here ('batch_size=1, # run on the slice of a single object'),
        because a conditioning frame can hold a different number of clicks per object. A frame that
        conditions nothing takes no prompts at all, and those are all but one frame per object, so
        there they can go through the model together.

        This is what volumetric prompt generation costs its time on, and it is bound by kernel
        launches rather than by arithmetic: sixteen objects on one slice issue sixteen forward passes
        of a few microseconds' worth of work each. Batching them leaves every mask exactly as it was
        - the objects carry no non-overlap constraint, so none of them depends on its batch - while
        the launches per frame drop by the size of the group.
        """
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        if start_frame_idx is None:
            start_frame_idx = min(
                frame_idx
                for output_dict in inference_state["output_dict_per_obj"].values()
                for frame_idx in output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            # Nothing to track backwards from the first frame.
            processing_order = range(start_frame_idx, end_frame_idx - 1, -1) if start_frame_idx > 0 else []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            to_track = []
            for obj_idx in range(batch_size):
                output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # A frame this object was prompted on already holds its output, see SAM2.
                if frame_idx in output_dict["cond_frame_outputs"]:
                    current_out = output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks_per_obj[obj_idx] = current_out["pred_masks"].to(
                        inference_state["device"], non_blocking=True
                    )
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(inference_state, frame_idx, obj_idx)
                    inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                else:
                    to_track.append(obj_idx)

            if to_track:
                # '_BatchedMemory' concatenates offloaded entries on the host.
                self._wait_for_offloaded_state(inference_state)
            for group in self._memory_groups(inference_state, to_track):
                pred_masks = self._track_frame_batch(inference_state, group, frame_idx, reverse)
                for index, obj_idx in enumerate(group):
                    pred_masks_per_obj[obj_idx] = pred_masks[index:index + 1]
                    inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}

            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
            yield frame_idx, obj_ids, video_res_masks

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


def _build_sam2_video_predictor(config_file, ckpt_path=None, device="cuda"):
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    hydra_overrides = [
        "++model._target_=micro_sam.v2.models._video_predictor.CustomVideoPredictor",
    ]

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model
