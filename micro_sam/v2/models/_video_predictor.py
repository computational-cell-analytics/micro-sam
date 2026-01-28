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
    if isinstance(img_path, str):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        video_width, video_height = img_pil.size  # the original video size
    else:
        img_np = img_path
        img_np = np.stack([img_np] * 3, axis=-1) if img_np.ndim == 2 else img_np  # Make it in RGB style.
        img_np = resize(
            img_np,
            output_shape=(image_size, image_size, 3),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ).astype(img_np.dtype)
        video_height, video_width = img_path.shape

    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")

    img = torch.from_numpy(img_np).permute(2, 0, 1)
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
    """

    @torch.inference_mode()
    def init_state(
        self,
        volume: np.ndarray,
        volume_embeddings: Optional[np.ndarray] = None,
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
            video_path=None,  # NOTE: The feature works. We just don't care about it in our tasks.
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
        }

        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}

        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}

        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []

        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}

        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}

        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}

        # visual features on a small number of recently visited frames for quick interactions
        if "cached_features" not in inference_state:  # The key does not exists in running 'inference_state'
            inference_state["cached_features"] = {}

        if ignore_caching_features:
            return inference_state

        # The backbone here computes the image embeddings (and other necessary things), if not provied
        def _get_all_backbone_out(frame_idx):
            image = images[frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            running_features = inference_state.get("cached_features") or {}
            running_features[frame_idx] = (image, backbone_out)
            inference_state["cached_features"] = running_features

        if volume_embeddings is None:  # Well, we compute the embeddings here on the fly.
            [_get_all_backbone_out(i) for i in range(inference_state["num_frames"])]

        else:  # Precomputed embeddings exists. Just provide the expected stuff to inference state.
            feats = volume_embeddings["features"]
            pos_list = volume_embeddings["pos_enc"]
            fpn_list = volume_embeddings["fpn"]
            running_features = inference_state.get("cached_features") or {}

            for frame_idx in range(inference_state["num_frames"]):
                image = images[frame_idx].to(device).float().unsqueeze(0)

                vision_features = torch.as_tensor(np.asarray(feats[frame_idx]), device=device).float()
                vision_pos_enc = [torch.as_tensor(np.asarray(t[frame_idx]), device=device).float() for t in pos_list]
                backbone_fpn = [torch.as_tensor(np.asarray(t[frame_idx]), device=device).float() for t in fpn_list]
                backbone_out = {
                    "vision_features": vision_features, "vision_pos_enc": vision_pos_enc, "backbone_fpn": backbone_fpn,
                }
                running_features[frame_idx] = (image, backbone_out)

            inference_state["cached_features"] = running_features

        return inference_state


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
