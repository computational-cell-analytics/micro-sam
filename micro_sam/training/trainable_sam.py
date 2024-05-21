import math
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
# from torch.nn.parameter import Parameter

from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide


# simple wrapper around SAM in order to keep things trainable
class TrainableSAM(nn.Module):
    """Wrapper to make the SegmentAnything model trainable.

    Args:
        sam: The SegmentAnything Model.
        device: The device for training.
    """
    def __init__(
        self,
        sam: Sam,
        device: Union[str, torch.device],
    ) -> None:
        super().__init__()
        self.sam = sam
        self.device = device
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)

    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Resize, normalize pixel values and pad to a square input.

        Args:
            x: The input tensor.

        Returns:
            The resized, normalized and padded tensor.
            The shape of the image after resizing.
        """

        # Resize longest side to match the image encoder.
        x = self.transform.apply_image_torch(x)
        input_size = x.shape[-2:]

        # Normalize colors
        x = (x - self.sam.pixel_mean.unsqueeze(0)) / self.sam.pixel_std.unsqueeze(0)

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x, input_size

    def image_embeddings_oft(self, batched_inputs):
        # Compute the input images.
        input_images, input_size = self.preprocess(
            torch.stack([x["image"] for x in batched_inputs], dim=0).to(self.device)
        )
        # Update the input size for each input in the batch.
        for i in range(len(batched_inputs)):
            batched_inputs[i]["input_size"] = input_size
        # Compute the image embeddings.
        image_embeddings = self.sam.image_encoder(input_images)
        return image_embeddings, batched_inputs

    # batched inputs follow the same syntax as the input to sam.forward
    def forward(
        self,
        batched_inputs: List[Dict[str, Any]],
        image_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> List[Dict[str, Any]]:
        """Forward pass.

        Args:
            batched_inputs: The batched input images and prompts.
            image_embeddings: The precompute image embeddings. If not passed then they will be computed.
            multimask_output: Whether to predict mutiple or just a single mask.

        Returns:
            The predicted segmentation masks and iou values.
        """
        outputs = []
        for image_record, curr_embedding in zip(batched_inputs, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"].to(self.device), image_record["point_labels"].to(self.device))
            else:
                points = None

            if "boxes" in image_record:
                boxes = image_record.get("boxes").to(self.device)
            else:
                boxes = None

            if "mask_inputs" in image_record:
                masks = image_record.get("mask_inputs").to(self.device)
            else:
                masks = None

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=image_record["input_size"],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "low_res_masks": low_res_masks,
                    "masks": masks,
                    "iou_predictions": iou_predictions
                }
            )

        return outputs


class LoRA_qkv(nn.Module):
    """Inspired from: https://github.com/JamesQFreeman/Sam_LoRA/

    In SAM, it is implemented as:
    ```python
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    ```
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B, N, N, 3 * org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


# TODO: the mask decoder has some attention blocks, need to decide if we perform lora on them as well.
# reference: SAMed and Maceij-SAM performs these experiments.
class LoRA_Sam(nn.Module):
    """Inspired from: https://github.com/JamesQFreeman/Sam_LoRA/

    Applies low-rank adaptation to the Segment Anything model's image encoder.

    Args:
        sam_model: a vision transformer model.
        rank: rank of LoRA.
        lora_layer: which specific layers we apply LoRA to.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, rank=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(
        self,
        sam_model: Sam,
        rank: int,
        lora_layer=None
    ):
        super(LoRA_Sam, self).__init__()

        assert rank > 0

        if lora_layer:
            self.lora_layer = lora_layer
        else:   # Only apply lora to the image encoder by default
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks))
            )

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # let's freeze all the image encoder layers first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_v = nn.Linear(rank, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output):
        return self.sam(batched_input, multimask_output)

    # TODO: the codebase below is not required here (part of original LoRA implementation)
    # we should port the relevant parts from below to `get_sam_model` (if required) for loading the model.

    # def save_lora_parameters(self, filename: str) -> None:
    #     """Only safetensors is supported now.

    #     Please install safetensor using: `pip install safetensor`, if you do not have one installed yet.

    #     This function saves both lora and fc parameters.
    #     """

    #     assert filename.endswith(".pt") or filename.endswith('.pth')

    #     num_layer = len(self.w_As)  # actually, it is half
    #     a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
    #     b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
    #     prompt_encoder_tensors = {}
    #     mask_decoder_tensors = {}

    #     # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
    #     if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
    #         self.sam, torch.nn.parallel.DistributedDataParallel
    #     ):
    #         state_dict = self.sam.module.state_dict()
    #     else:
    #         state_dict = self.sam.state_dict()

    #     for key, value in state_dict.items():
    #         if 'prompt_encoder' in key:
    #             prompt_encoder_tensors[key] = value
    #         if 'mask_decoder' in key:
    #             mask_decoder_tensors[key] = value

    #     merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
    #     torch.save(merged_dict, filename)

    # def load_lora_parameters(self, filename: str) -> None:
    #     r"""Only safetensors is supported now.

    #     Please install safetensor using: `pip install safetensor`, if you do not have one installed yet.

    #     This function loads both lora and fc parameters.
    #     """

    #     assert filename.endswith(".pt") or filename.endswith('.pth')

    #     state_dict = torch.load(filename)

    #     for i, w_A_linear in enumerate(self.w_As):
    #         saved_key = f"w_a_{i:03d}"
    #         saved_tensor = state_dict[saved_key]
    #         w_A_linear.weight = Parameter(saved_tensor)

    #     for i, w_B_linear in enumerate(self.w_Bs):
    #         saved_key = f"w_b_{i:03d}"
    #         saved_tensor = state_dict[saved_key]
    #         w_B_linear.weight = Parameter(saved_tensor)

    #     sam_dict = self.sam.state_dict()
    #     sam_keys = sam_dict.keys()

    #     # load prompt encoder
    #     prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
    #     prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
    #     prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
    #     sam_dict.update(prompt_encoder_new_state_dict)

    #     # load mask decoder
    #     mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
    #     mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
    #     mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
    #     sam_dict.update(mask_decoder_new_state_dict)
    #     self.sam.load_state_dict(sam_dict)
