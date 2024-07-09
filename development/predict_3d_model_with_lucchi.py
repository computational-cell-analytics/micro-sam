import os
import argparse
from tqdm import tqdm
import numpy as np
import imageio.v3 as imageio
from elf.io import open_file
from skimage.measure import label as connected_components

import torch
from glob import glob

from torch_em.util.segmentation import size_filter
from torch_em.util import load_model
from torch_em.transform.raw import normalize
from torch_em.util.prediction import predict_with_halo

from micro_sam import util
from micro_sam.evaluation.inference import _run_inference_with_iterative_prompting_for_image

from segment_anything import SamPredictor

from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
from typing import List, Union, Dict, Optional, Tuple


class RawTrafoFor3dInputs:
    def _normalize_inputs(self, raw):
        raw = normalize(raw)
        raw = raw * 255
        return raw

    def _set_channels_for_inputs(self, raw):
        raw = np.stack([raw] * 3, axis=0)
        return raw

    def __call__(self, raw):
        raw = self._normalize_inputs(raw)
        raw = self._set_channels_for_inputs(raw)
        return raw


def _run_semantic_segmentation_for_image_3d(
    model: torch.nn.Module,
    image: np.ndarray,
    prediction_path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    halo: Tuple[int, int, int],
):
    device = next(model.parameters()).device
    block_shape = tuple(bs - 2 * ha for bs, ha in zip(patch_shape, halo))

    def preprocess(x):
        x = 255 * normalize(x)
        x = np.stack([x] * 3)
        return x

    def prediction_function(net, inp):
        # Note: we have two singleton axis in front here, I am not quite sure why.
        # Both need to be removed to be compatible with the SAM network.
        batched_input = [{
            "image": inp[0, 0], "original_size": inp.shape[-2:]
        }]
        masks = net(batched_input, multimask_output=True)[0]["masks"]
        masks = torch.argmax(masks, dim=1)
        return masks

    # num_classes = model.sam_model.mask_decoder.num_multimask_outputs
    image_size = patch_shape[-1]
    output = np.zeros(image.shape, dtype="float32")
    predict_with_halo(
        image, model, gpu_ids=[device],
        block_shape=block_shape, halo=halo,
        preprocess=preprocess, output=output,
        prediction_function=prediction_function
    )

    # save the segmentations
    imageio.imwrite(prediction_path, output, compression="zlib")


def run_semantic_segmentation_3d(
    model: torch.nn.Module,
    image_paths: List[Union[str, os.PathLike]],
    prediction_dir: Union[str, os.PathLike],
    semantic_class_map: Dict[str, int],
    patch_shape: Tuple[int, int, int] = (32, 512, 512),
    halo: Tuple[int, int, int] = (6, 64, 64),
    image_key: Optional[str] = None,
    is_multiclass: bool = False,
):
    """
    """
    for image_path in tqdm(image_paths, desc="Run inference for semantic segmentation with all images"):
        image_name = os.path.basename(image_path)

        assert os.path.exists(image_path), image_path

        # Perform segmentation only on the semantic class
        # for i, (semantic_class_name, _) in enumerate(semantic_class_map.items()):
        #     if is_multiclass:
        #         semantic_class_name = "all"
        #         if i > 0:  # We only perform segmentation for multiclass once.
        #             continue

        semantic_class_name = "all" #since we only perform segmentation for multiclass
            # We skip the images that already have been segmented
        image_name = os.path.splitext(image_name)[0] + ".tif"
        prediction_path = os.path.join(prediction_dir, "all", image_name)
        if os.path.exists(prediction_path):
            continue

        if image_key is None:
            image = imageio.imread(image_path)
        else:
            with open_file(image_path, "r") as f:
                image = f[image_key][:]

        # create the prediction folder
        os.makedirs(os.path.join(prediction_dir, semantic_class_name), exist_ok=True)

        _run_semantic_segmentation_for_image_3d(
            model=model, image=image, prediction_path=prediction_path,
            patch_shape=patch_shape, halo=halo,
        )


def transform_labels(y):
    return (y > 0).astype("float32")


def predict(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint_path is not None:
        if os.path.exists(args.checkpoint_path):
            # model = load_model(checkpoint=args.checkpoint_path, device=device) # does not work
            
            cp_path = os.path.join(args.checkpoint_path, "", "best.pt")
            print(cp_path)
            model = get_sam_3d_model(device, n_classes=args.n_classes, image_size=args.patch_shape[1],
                                     lora_rank=4,
                                     model_type=args.model_type,
                                     # checkpoint_path=args.checkpoint_path
                                     ) 
            
            checkpoint = torch.load(cp_path, map_location=device)
            # # Load the state dictionary from the checkpoint
            for k, v in checkpoint.items():
                print("keys", k)
            model.load_state_dict(checkpoint['model_state']) #.state_dict()
            model.eval()

    data_paths = glob(os.path.join(args.input_path, "**/*test.h5"), recursive=True)
    pred_path = args.save_root
    semantic_class_map = {"all": 0}
            
    run_semantic_segmentation_3d(
        model=model, image_paths=data_paths, prediction_dir=pred_path, semantic_class_map=semantic_class_map,
        patch_shape=args.patch_shape, image_key="raw", is_multiclass=True
    )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LiveCELL dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/projects/nim00007/sam/data/lucchi/",
        help="The filepath to the LiveCELL data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 512, 512), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--n_classes", type=int, default=3, help="Number of classes to predict")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument(
        "--save_root", "-s", default="/scratch-grete/usr/nimlufre/micro-sam3d",
        help="The filepath to where the logs and the checkpoints will be saved."
    )
    parser.add_argument(
        "--checkpoint_path", "-c", default="/scratch-grete/usr/nimlufre/micro-sam3d/checkpoints/3d-sam-vitb-masamhyp-lucchi",
        help="The filepath to where the logs and the checkpoints will be saved."
    )

    args = parser.parse_args()
    
    predict(args)


if __name__ == "__main__":
    main()
