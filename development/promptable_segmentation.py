import os
from glob import glob
from natsort import natsorted
from typing import Union, Literal, Optional, Tuple, List

import json
import numpy as np
import imageio.v3 as imageio
from skimage.measure import regionprops
from skimage.measure import label as connected_components

from torch_em.data.datasets.util import download_source, unzip

from micro_sam.util import get_sam_model, precompute_image_embeddings
from micro_sam.prompt_based_segmentation import segment_from_points, segment_from_box


URL = "https://api.data.neurosys.com:4443/agar-public/AGAR_demo.zip"
ROOT = "data"  # override this to your desired folder to store the data.


def get_agar_data(path: Union[os.PathLike, str] = ROOT) -> str:
    data_dir = os.path.join(path, "agar")
    if os.path.exists(os.path.join(data_dir, "AGAR_representative")):
        return os.path.join(data_dir, "AGAR_representative")

    os.makedirs(os.path.join(path, "agar"), exist_ok=True)

    # Download the dataset
    zip_path = os.path.join(data_dir, "AGAR_demo.zip")
    download_source(path=zip_path, url=URL, download=True)
    unzip(zip_path=zip_path, dst=data_dir)

    return os.path.join(data_dir, "AGAR_representative")


def get_agar_paths(
    path: Union[os.PathLike, str] = ROOT, resolution: Optional[Literal["higher", "lower"]] = None,
) -> Tuple[List[str], List[str]]:

    data_dir = get_agar_data(path)

    # Get path to one low-res image and corresponding metadata file.
    resolution = ("*" if resolution is None else resolution) + "-resolution"
    image_paths = natsorted(glob(os.path.join(data_dir, resolution, "*.jpg")))
    metadata_paths = [p.replace(".jpg", ".json") for p in image_paths]
    metadata_paths = [p for p in metadata_paths if os.path.exists(p)]

    assert image_paths and len(image_paths) == len(metadata_paths)

    return image_paths, metadata_paths


def extract_prompts_from_data(prompt_choice: Literal["points", "box"] = "points") -> Tuple[List[np.ndarray], List]:

    if prompt_choice not in ["points", "box"]:
        raise ValueError(f"'{prompt_choice}' is not a valid prompt choice. Choose from 'points' / 'box'.")

    # Get the inputs
    image_paths, metadata_paths = get_agar_paths(resolution="lower")

    images = [imageio.imread(p) for p in image_paths]
    prompts_per_image = []
    for i, metadata_path in enumerate(metadata_paths):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if prompt_choice == "points":
            # Extract point coordinates at the center of each labeled object.
            point_coords = [
                (label["y"] + int(label["height"] / 2), label["x"] + int(label["width"] / 2))
                for label in metadata["labels"]
            ]
            prompts_per_image.append(point_coords)

        else:
            # Extract bounding box coordinates for each labeled object.
            # NOTE: We get the specific format below to enable visualization in 'napari'.
            box_coords = [
                np.array(
                    [
                        [label["y"], label["x"]],
                        [label["y"], label["x"] + label["width"]],
                        [label["y"] + label["height"], label["x"] + label["width"]],
                        [label["y"] + label["height"], label["x"]]
                    ]
                ) for label in metadata["labels"]
            ]
            prompts_per_image.append(box_coords)

    return images, prompts_per_image


def main():
    # Make choice for a prompt
    prompt_choice = "box"
    view = True  # Whether to view the segmentations.

    # Get the images and corresponding point prompts.
    images, prompts_per_image = extract_prompts_from_data(prompt_choice)

    # Segment using point prompts.
    predictor = get_sam_model(model_type="vit_b_lm")

    for i, (image, prompts) in enumerate(zip(images, prompts_per_image)):

        # Compute the image embeddings.
        image_embeddings = precompute_image_embeddings(
            predictor=predictor,
            input_=image,
            ndim=2,  # With RGB images, we should have channels last and must set ndim to 2.
            verbose=False,
            # tile_shape=(384, 384),  # Tile shape for larger images.
            # halo=(64, 64),  # Overlap shape for larger images.
            # save_path=f"embeddings_{i}.zarr",  # Caches the image embeddings.
        )

        # Run promptable segmentation.
        if prompt_choice == "points":
            masks = [
                segment_from_points(
                    predictor=predictor,
                    points=np.array([point_coord]),  # Each point coordinate (Y, X) is expected as array.
                    labels=np.array([1]),  # Each corresponding label, eg. 1 corresponds positive, is expected as array.
                    image_embeddings=image_embeddings,
                ).squeeze() for point_coord in prompts
            ]

        elif prompt_choice == "box":
            # Extract box prompts in desired format, i.e. YXY'X', where ' represents max and others min values.
            box_prompts = [
                (np.min(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 0]), np.max(box[:, 1])) for box in prompts
            ]

            masks = [
                segment_from_box(
                    predictor=predictor,
                    box=np.array(box),  # Each bounding box is expected as array.
                    image_embeddings=image_embeddings,
                ).squeeze() for box in box_prompts
            ]

        else:
            raise NotImplementedError(f"The prompt choice '{prompt_choice}' is not valid.")

        # Merge all segmentations into one.

        # 1. First, we get the area per object and try to map as: big objects first and small ones then
        #    (to avoid losing tiny objects near-by or to overlaps)
        mask_props = [{"mask": mask, "area": regionprops(connected_components(mask))[0].area} for mask in masks]

        # 2. Next, we assort based on area from greatest to smallest.
        assorted_masks = sorted(mask_props, key=(lambda x: x["area"]), reverse=True)
        masks = [per_mask["mask"] for per_mask in assorted_masks]

        # 3. Finally, we merge all individual segmentations into one.
        segmentation = np.zeros(image.shape[:2], dtype=int)
        for j, mask in enumerate(masks, start=1):
            segmentation[mask > 0] = j

        if view:
            # Visualize the image and corresponding segmentation (and prompts).
            import napari
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(segmentation)

            if prompt_choice == "points":
                v.add_points(prompts)
            else:
                v.add_shapes(prompts, face_color="transparent", edge_width=2)

            napari.run()


if __name__ == "__main__":
    main()
