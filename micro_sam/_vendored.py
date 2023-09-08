"""
Functions from other third party libraries.

We can remove these functions once the bugs affecting our code is fixed upstream.

The license type of the thrid party software project must be compatible with
the software license the micro-sam project is distributed under.
"""
from typing import Any, Dict, List

import numpy as np
import torch


# segment_anything/util/amg.py
# https://github.com/facebookresearch/segment-anything
def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    assert masks.dtype == torch.bool

    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, dtype=torch.int, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    in_height_coords = in_height_coords.type(torch.int)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, dtype=torch.int, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    in_width_coords = in_width_coords.type(torch.int)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


# segment_anything/util/amg.py
# https://github.com/facebookresearch/segment-anything
def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """Calculates the runlength encoding of binary input masks.

    Implementation based on
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
    """
    # Put in fortran order and flatten h, w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)
    tensor = tensor.detach().cpu().numpy()

    n = tensor.shape[1]

    # encode the rle for the individual masks
    out = []
    for mask in tensor:
        diffs = mask[1:] != mask[:-1]  # pairwise unequal (string safe)
        indices = np.append(np.where(diffs), n - 1)  # must include last element position
        # count needs to start with 0 if the mask begins with 1
        counts = [] if mask[0] == 0 else [0]
        # compute the actual RLE
        counts += np.diff(np.append(-1, indices)).tolist()
        out.append({"size": [h, w], "counts": counts})
    return out
