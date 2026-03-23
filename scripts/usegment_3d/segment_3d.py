from tqdm import tqdm
from functools import partial

import numpy as np

import segment3D.usegment3d as uSegment3D
import segment3D.parameters as uSegment3D_params


def run_usegment3d_with_microsam(volume):
    # Run MicroSAM on 3d volume along all three directions.
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

    # Load the MicroSAM model.
    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm", segmentation_mode="ais")

    # Run AIS on our volume in a per-slice fashion.
    seg_runner = partial(
        automatic_instance_segmentation, predictor=predictor, segmenter=segmenter, ndim=2, verbose=False
    )

    instances_xy = np.stack(
        [seg_runner(input_path=curr_slice) for curr_slice in tqdm(volume, desc="xy")]
    )
    instances_xz = np.stack(
        [seg_runner(input_path=curr_slice) for curr_slice in tqdm(volume.transpose(1, 0, 2), desc="xz")]
    )
    # skip 'yz' for now
    instances_yz = []

    # Get the default parameters first.
    params = uSegment3D_params.get_2D_to_3D_aggregation_params()

    # The available choices are "cellpose_improve", "fmm", "cellpose_skel", "fmm_skel", "edt".
    params["indirect_method"]["dtform_method"] = "cellpose_improve"

    # And a few other parameters.
    params["indirect_method"]["dtform_method"] = "cellpose_improve"
    params["gradient_descent"]["gradient_decay"] = None
    params["indirect_method"]["smooth_skel_sigma"] = None
    params["indirect_method"]["edt_fixed_point_percentile"] = None

    # Run the uSegment3d's 'indirect' method for the most amount of flexibility.
    segmentation_3d, _ = uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(
        segmentations=[instances_xy, instances_xz, instances_yz],
        img_xy_shape=instances_xy.shape,   # full 3D shape, not one slice
        precomputed_binary=None,  # Seems like binary segmentations in xy direction.
        params=params,
        savefolder=None,
        basename="usegment3d_indirect_test",
    )

    # Store segmentations
    import h5py
    with h5py.File("test.h5", "w") as f:
        f.create_dataset("raw", data=volume, compression="gzip")
        f.create_dataset("segmentation/usegment3d-microsam", data=segmentation_3d, compression="gzip")


def main():
    # Let's work with the 'cell3d' example data in scikit-image.
    from skimage.data import cells3d
    volume = cells3d()[:, 0]  # input has shape of (60, 256, 256).
    run_usegment3d_with_microsam(volume)


if __name__ == "__main__":
    main()
