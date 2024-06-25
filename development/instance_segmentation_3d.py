import napari
from elf.io import open_file
import h5py
import os
import torch
import numpy as np

import micro_sam.sam_3d_wrapper as sam_3d
import micro_sam.util as util
# from micro_sam.segment_instances import (
#     segment_instances_from_embeddings,
#     segment_instances_sam,
#     segment_instances_from_embeddings_3d,
# )
from micro_sam import multi_dimensional_segmentation as mds
from micro_sam.visualization import compute_pca
INPUT_PATH_CLUSTER = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo/outer-membrane1/1_20230125_TOMO_HOI_WT_36859_J2_upSTEM750_BC3.6/upSTEM750_36859_J2_TS_SP_003_rec_2kb1dawbp_crop.h5"
# EMBEDDINGS_PATH_CLUSTER = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo/outer-membrane1/1_20230125_TOMO_HOI_WT_36859_J2_upSTEM750_BC3.6/embedding-mito-3d.zarr"
EMBEDDINGS_PATH_CLUSTER = "/scratch-grete/usr/nimlufre/"
INPUT_PATH_LOCAL = "/home/freckmann15/data/mitochondria/cooper/mito_tomo/outer-membrane1/1_20230125_TOMO_HOI_WT_36859_J2_upSTEM750_BC3.6/upSTEM750_36859_J2_TS_SP_003_rec_2kb1dawbp_crop.h5"
EMBEDDINGS_PATH_LOCAL = "/home/freckmann15/data/mitochondria/cooper/mito_tomo/outer-membrane1/1_20230125_TOMO_HOI_WT_36859_J2_upSTEM750_BC3.6/"
INPUT_PATH = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/volume_em/training_blocks_v1/4007_cutout_1.h5"
EMBEDDINGS_PATH = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/volume_em/training_blocks_v1/embedding-mito-3d.zarr"
TIMESERIES_PATH = "../examples/data/DIC-C2DH-HeLa/train/01"
EMBEDDINGS_TRACKING_PATH = "../examples/embeddings/embeddings-ctc.zarr"

# def cell_segmentation_3d() -> None:
#     with open_file(TIMESERIES_PATH, mode="r") as f:
#         timeseries = f["*.tif"][:50]
    
#     predictor = util.get_sam_model()
#     image_embeddings = util.precompute_image_embeddings(predictor, timeseries, EMBEDDINGS_TRACKING_PATH)

#     seg = segment_instances_from_embeddings_3d(predictor, image_embeddings)

#     v = napari.Viewer()
#     v.add_image(timeseries)
#     v.add_labels(seg)
#     napari.run()


# def _get_dataset_and_reshape(path: str, key: str = "raw", shape: tuple = (32, 256, 256)) -> np.ndarray:

#     with h5py.File(path, "r") as f:
#         # Check if the key exists in the file
#         if key not in f:
#             raise KeyError(f"Dataset with key '{key}' not found in file '{path}'.")

#         # Load the dataset
#         dataset = f[key][...]

#     # Reshape the dataset
#     if dataset.shape != shape:
#         try:
#             # Attempt to reshape the dataset to the desired shape
#             dataset = dataset.reshape(shape)
#         except ValueError:
#             raise ValueError(f"Failed to reshape dataset with key '{key}' to shape {shape}.")

#     return dataset
def get_dataset_cutout(path: str, key: str = "raw", shape: tuple = (32, 256, 256),
                       start_index: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Loads a cutout from a dataset in an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        key (str, optional): Key of the dataset to load. Defaults to "raw".
        shape (tuple, optional): Desired shape of the cutout. Defaults to (32, 256, 256).
        start_index (tuple, optional): Starting index for the cutout within the dataset.
            Defaults to None, which selects a random starting point within valid bounds.

    Returns:
        np.ndarray: The loaded cutout of the dataset with the specified shape.

    Raises:
        KeyError: If the specified key is not found in the HDF5 file.
        ValueError: If the cutout shape exceeds the dataset dimensions or the starting index is invalid.
    """

    with h5py.File(path, "r") as f:

        dataset = f[key]
        dataset_shape = dataset.shape
        print("original data shape", dataset_shape)

        # Validate cutout shape
        if any(s > d for s, d in zip(shape, dataset_shape)):
            raise ValueError(f"Cutout shape {shape} exceeds dataset dimensions {dataset_shape}.")

        # Generate random starting index if not provided
        if start_index is None:
            start_index = tuple(np.random.randint(0, dim - s + 1, size=len(shape)) for dim, s in zip(dataset_shape, shape))

        # Calculate end index
        end_index = tuple(min(i + s, dim) for i, s, dim in zip(start_index, shape, dataset_shape))

        # Load the cutout
        cutout = dataset[start_index[0]:end_index[0],
                         start_index[1]:end_index[1],
                         start_index[2]:end_index[2]]
        print("cutout data shape", cutout.shape)

    return cutout


def mito_segmentation_3d() -> None:
    patch_shape = (32, 256, 256)
    start_index = (10, 32, 64)
    data_slice = get_dataset_cutout(INPUT_PATH_LOCAL, shape=patch_shape)  #start_index=start_index

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_b"
    predictor, sam = util.get_sam_model(return_sam=True, model_type=model_type, device=device)

    d_size = 3
    predictor3d = sam_3d.Predictor3D(sam, d_size)
    print(predictor3d)
    #breakpoint()
    predictor3d.model.forward(torch.from_numpy(data_slice), multimask_output=False, image_size=patch_shape)
    # output = predictor3d.model([data_slice], multimask_output=False)#image_size=patch_shape

    # predictor3d._hash = util.models().registry[model_type]

    # predictor3d.model_name = model_type

    # image_embeddings = util.precompute_image_embeddings(predictor3d, volume, EMBEDDINGS_PATH_CLUSTER)
    # seg = util.segment_instances_from_embeddings_3d(predictor3d, image_embeddings)
    
    # prediction_filename = os.path.join(EMBEDDINGS_PATH_CLUSTER, f"prediction_{INPUT_PATH_CLUSTER}.h5")
    # with h5py.File(prediction_filename, "w") as prediction_file:
    #     prediction_file.create_dataset("prediction", data=seg)

    # visualize
    # v = napari.Viewer()
    # v.add_image(volume)
    # v.add_labels(seg)
    # v.add_labels(seg_sam)
    # napari.run()
    


def main():
    # automatic segmentation for the data from Lucchi et al. (see 'sam_annotator_3d.py')
    # nucleus_segmentation(use_mws=True)
    mito_segmentation_3d()

    # automatic segmentation for data from the cell tracking challenge (see 'sam_annotator_tracking.py')
    # cell_segmentation(use_mws=True)
    # cell_segmentation_3d()


if __name__ == "__main__":
    main()
