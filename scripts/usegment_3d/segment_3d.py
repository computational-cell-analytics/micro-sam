import logging
from tqdm import tqdm
from functools import partial

import h5py
import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.light_microscopy.embedseg_data import get_embedseg_paths

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def run_microsam3d(volume, labels=None, save_path="test.h5"):
    # Load the MicroSAM model.
    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm", segmentation_mode="ais")

    # Run AIS on our entire volume.
    instances = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, ndim=3, input_path=volume,
    )

    # Store segmentations
    with h5py.File(save_path, "a") as f:
        if "raw" not in f:
            f.create_dataset("raw", data=volume, compression="gzip")
        if "labels" not in f and labels is not None:
            f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("segmentation/microsam3d", data=instances, compression="gzip")


def run_usegment3d_with_microsam(volume, labels=None, save_path="test.h5"):
    # Run MicroSAM on 3d volume along all three directions.
    # NOTE: Install uSegment3D using `pip install u-Segment3D`.
    import segment3D.usegment3d as uSegment3D
    import segment3D.parameters as uSegment3D_params

    # Suppress noisy Dask distributed worker/nanny startup logs.
    for _logger in (
        "distributed", "distributed.nanny", "distributed.worker", "distributed.scheduler", "distributed.core"
    ):
        logging.getLogger(_logger).setLevel(logging.WARNING)

    # Load the MicroSAM model.
    predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm", segmentation_mode="ais")

    # Run AIS on our volume in a per-slice fashion.
    seg_runner = partial(
        automatic_instance_segmentation, predictor=predictor, segmenter=segmenter, ndim=2, verbose=False
    )

    instances_xy = np.stack(
        [seg_runner(input_path=curr_slice) for curr_slice in tqdm(volume, desc="Segment XY")]
    )
    instances_xz = np.stack(
        [seg_runner(input_path=curr_slice) for curr_slice in tqdm(volume.transpose(1, 0, 2), desc="Segment XZ")]
    )
    instances_yz = np.stack(
        [seg_runner(input_path=curr_slice) for curr_slice in tqdm(volume.transpose(2, 0, 1), desc="Segment YZ")]
    )

    # Get the default parameters first.
    params = uSegment3D_params.get_2D_to_3D_aggregation_params()
    params["indirect_method"]["n_cpu"] = 4  # default spawns '(cpu_count - 1) // 2 dask workers'

    # The available choices are "cellpose_improve", "fmm", "cellpose_skel", "fmm_skel", "edt".
    params["indirect_method"]["dtform_method"] = "cellpose_improve"

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
    with h5py.File(save_path, "a") as f:
        if "raw" not in f:
            f.create_dataset("raw", data=volume, compression="gzip")
        if "labels" not in f and labels is not None:
            f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("segmentation/usegment3d-microsam", data=segmentation_3d, compression="gzip")


def evaluate_results(save_path):
    with h5py.File(save_path, "r") as f:
        labels = f["labels"][:]
        seg_microsam3d = f["segmentation/microsam3d"][:]
        seg_microsam_usegment3d = f["segmentation/usegment3d-microsam"][:]

    # Let's evaluate and see how the results are
    from elf.evaluation import mean_segmentation_accuracy
    msa_microsam3d = mean_segmentation_accuracy(labels, seg_microsam3d)
    msa_microsam_usegment3d = mean_segmentation_accuracy(labels, seg_microsam_usegment3d)
    print(msa_microsam3d, msa_microsam_usegment3d)


def main():
    # Let's work with the 'cell3d' example data in scikit-image.
    # from skimage.data import cells3d
    # volume = cells3d()[:, 1]  # input has shape of (60, 256, 256).

    # Let's work with labeled data (so that we can evaluate)
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/embedseg"
    raw_paths, label_paths = get_embedseg_paths(
        path=data_dir, name="Mouse-Skull-Nuclei-CBG", split="test",
    )
    volume, labels = imageio.imread(raw_paths[0]), imageio.imread(label_paths[0])

    # Run / evaluate segmentation models.
    save_path = "embedseg_mouse_skull_nuclei.h5"

    run_usegment3d_with_microsam(volume, labels, save_path)
    run_microsam3d(volume, labels, save_path)

    # Observations on Mouse-Skull-Nuclei-CBG data:
    # MicroSAM3d: 0.359 | MicroSAM2d + uSegment3d: 0.479

    evaluate_results(save_path)


if __name__ == "__main__":
    main()
