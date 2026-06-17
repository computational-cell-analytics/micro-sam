import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Optional, Dict, Union

import numpy as np
from PIL import Image
from skimage.transform import resize

import torch

from sam2.build_sam import _load_checkpoint
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import AsyncVideoFrameLoader


def _load_img_as_tensor(img_path, image_size):
    """Load a single frame as a float32 [0, 1] tensor of shape (3, image_size, image_size).

    For file-path inputs: PIL loads the image, resizes via plain square stretch (JPEG convention).
    For numpy inputs: percentile-normalizes any dtype to [0, 1] (2nd / 98th percentile per channel);
    resizes using aspect-ratio preserving scale to image_size on the longest side, then zero-pads to a
    square - matching ConvertToSam2VideoBatch._to_sam2_size used during training.

    Returns:
        img: (3, image_size, image_size) float32 tensor, ImageNet-normalised by the caller.
        video_height: max(H, W) of the original frame - used as the effective square dimension
            for SAM2's coordinate normalization so prompts map into the resized content region.
        video_width: same as video_height.
    """
    if isinstance(img_path, str):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        video_width, video_height = img_pil.size
        img_np = img_np / 255.0
    else:
        img_np = img_path
        img_np = np.stack([img_np] * 3, axis=-1) if img_np.ndim == 2 else img_np

        # Percentile-normalize each channel to [0, 1], so any input dtype (e.g. uint16 microscopy data)
        # is mapped to the range SAM2's ImageNet normalization expects. Clip since percentile
        # normalization maps the 2nd / 98th percentiles to 0 / 1 and overshoots outside that range.
        from torch_em.transform.raw import normalize_percentile
        img_np = normalize_percentile(img_np.astype(np.float32), lower=2.0, upper=98.0, axis=(0, 1))
        img_np = np.clip(img_np, 0.0, 1.0)

        # Aspect-ratio preserving scale + zero-pad, matching _to_sam2_size in training.
        # video_height/video_width are set to max(H, W) so SAM2's coordinate normalization
        # (which divides by these and scales to image_size) correctly maps original-frame
        # coordinates into the resized content region rather than the zero-padded area.
        H, W = img_np.shape[:2]
        video_height = video_width = max(H, W)
        scale = image_size / max(H, W)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        img_np = resize(img_np, output_shape=(new_h, new_w, 3), order=1, anti_aliasing=True, preserve_range=True)
        pad_h, pad_w = image_size - new_h, image_size - new_w
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)))

    img = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
    return img, video_height, video_width


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
        assert isinstance(volume, np.ndarray) and volume.ndim == 3, "Something is off with the 'volume'."
        # Iterate over each slice.
        images = []
        for i, curr_slice in enumerate(volume):
            curr_image, video_height, video_width = _load_img_as_tensor(curr_slice, image_size)
            images.append(curr_image)
        images = torch.stack(images)  # Stack the inputs in expected format.
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

            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            vision_pos_enc = [
                torch.as_tensor(np.asarray(t[frame_idx]), device=device).float() for t in embeddings["pos_enc"]
            ]
            backbone_fpn = [
                torch.as_tensor(np.asarray(t[frame_idx]), device=device).float() for t in embeddings["fpn"]
            ]
            backbone_out = {"backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc}
            # Keep only the most recent frame's features, matching upstream SAM2's single-frame cache.
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

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
