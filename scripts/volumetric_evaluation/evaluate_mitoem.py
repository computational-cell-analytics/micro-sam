# in-domain case for electron microscopy generalist
#
# NOTE:
#   1. for the em organelles generalist, we use the following rois:
#       - for training = [np.s_[100:110, :, :], np.s_[100:110, :, :]]
#       - for validation = [np.s_[0:5, :, :], np.s_[0:5, :, :]]
#   2. for the grid search below: (shape: (50, 768, 768))
#       - for validation: we take the training set and sample one volume with most instances
#       - for testing: we take the validation set and sample one volume with most instances


import os

import h5py
import numpy as np
import pandas as pd
from skimage.measure import label

from elf.evaluation import mean_segmentation_accuracy

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_mitoem_loader

from micro_sam.training import identity
from micro_sam import instance_segmentation
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation
from micro_sam.evaluation.multi_dimensional_segmentation import run_multi_dimensional_segmentation_grid_search


def create_raw_and_label_volumes(data_path, species):
    def _get_volumes_from_loaders(split):
        if split == "train":
            rois = [np.s_[:100, :, :]]
        elif split == "val":
            rois = [np.s_[5:, :, :]]
        else:
            raise ValueError

        loader = get_mitoem_loader(
            path=data_path,
            splits=split,
            patch_shape=(50, 768, 768),
            batch_size=1,
            samples=[species],
            sampler=MinInstanceSampler(),
            raw_transform=identity,
            rois=rois,
            num_workers=16,
        )

        max_val = 0
        for x, y in loader:
            num_instances = len(np.unique(y))
            if max_val < num_instances:
                max_val = num_instances
                chosen_raw, chosen_labels = x, y

        return chosen_raw, chosen_labels

    _path_to_new_volume = os.path.join(data_path, "for_micro_sam", f"mitoem_{species}.h5")
    if os.path.exists(_path_to_new_volume):
        print("The volume is already computed and stored at:", _path_to_new_volume)
        return _path_to_new_volume
    else:
        os.makedirs(os.path.split(_path_to_new_volume)[0], exist_ok=True)

    print(f"Creating the volumes for {species}...")
    val_raw, val_labels = _get_volumes_from_loaders("train")
    test_raw, test_labels = _get_volumes_from_loaders("val")

    # now let's save them
    with h5py.File(_path_to_new_volume, "a") as f:
        f.create_dataset("volume/val/raw", data=val_raw.numpy().squeeze(), compression="gzip")
        f.create_dataset("volume/val/labels", data=val_labels.numpy().squeeze(), compression="gzip")
        f.create_dataset("volume/test/raw", data=test_raw.numpy().squeeze(), compression="gzip")
        f.create_dataset("volume/test/labels", data=test_labels.numpy().squeeze(), compression="gzip")

    print("The volume has been computed and stored at", _path_to_new_volume)
    return _path_to_new_volume


def get_raw_and_label_volumes(volume_path, split):
    with h5py.File(volume_path, "r") as f:
        raw = f[f"volume/{split}/raw"][:]
        labels = f[f"volume/{split}/labels"][:]

    assert raw.shape == labels.shape

    # applying connected components to get instances
    labels = label(labels)

    return raw, labels


def _3d_automatic_instance_segmentation_with_decoder(args, volume_path, species):
    # let's run this on ais
    test_raw, test_labels = get_raw_and_label_volumes(volume_path, "test")

    model_type = args.model_type
    checkpoint_path = args.checkpoint
    res_dir = os.path.join(args.resdir, "auto")
    res_path = os.path.join(res_dir, "ais.csv")
    if os.path.exists(res_dir):
        print("The results are already saved at:", res_path)
        return
    else:
        os.makedirs(res_dir, exist_ok=True)

    predictor, decoder = instance_segmentation.get_predictor_and_decoder(model_type, checkpoint_path)
    segmentor = instance_segmentation.InstanceSegmentationWithDecoder(predictor, decoder)

    auto_3d_seg_kwargs = {
        "center_distance_threshold": 0.3,
        "boundary_distance_threshold": 0.4,
        "distance_smoothing": 2.2,
        "min_size": 200,
        "gap_closing": 2,
        "min_z_extent": 2
    }

    instances = automatic_3d_segmentation(
        volume=test_raw,
        predictor=predictor,
        segmentor=segmentor,
        embedding_path=os.path.join(args.embedding_path, species, "test"),
        **auto_3d_seg_kwargs
    )

    msa, sa = mean_segmentation_accuracy(instances, test_labels, return_accuracies=True)
    print("mSA for 3d volume is:", msa)
    print("SA50 for 3d volume is:", sa[0])
    save_auto_res = {
        "mSA": msa, "SA50": sa[0]
    }
    save_auto_resdf = pd.DataFrame.from_dict([save_auto_res])
    save_auto_resdf.to_csv(res_path)


def _3d_interactive_instance_segmentation(args, volume_path, species):
    # let's do grid-search on the val set
    val_raw, val_labels = get_raw_and_label_volumes(volume_path, "val")

    best_params_path = run_multi_dimensional_segmentation_grid_search(
        volume=val_raw,
        ground_truth=val_labels,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        embedding_path=os.path.join(args.embedding_path, species, "val"),
        result_dir=os.path.join(args.resdir, species, "val"),
        interactive_seg_mode="box",
        verbose=False,
    )

    best_params = {}
    resdf = pd.read_csv(best_params_path)
    for k, v in resdf.loc[0].items():
        if k.startswith("Unnamed") or k == "mSA":
            continue
        best_params[k] = [v]

    # now let's use the best parameters on the test set
    test_raw, test_labels = get_raw_and_label_volumes(volume_path, "test")

    run_multi_dimensional_segmentation_grid_search(
        volume=test_raw,
        ground_truth=test_labels,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        embedding_path=os.path.join(args.embedding_path, species, "test"),
        result_dir=os.path.join(args.resdir, species, "test"),
        interactive_seg_mode="box",
        verbose=False,
        grid_search_values=best_params
    )


def for_one_species(args):
    volume_path = create_raw_and_label_volumes(args.input_path, args.species)

    if args.ais:
        _3d_automatic_instance_segmentation_with_decoder(args, volume_path, args.species)

    if args.int:
        _3d_interactive_instance_segmentation(args, volume_path, args.species)


def main(args):
    assert args.species is not None, "Choose from 'human' / 'rat'"
    for_one_species(args)


if __name__ == "__main__":
    from util import _get_default_args
    args = _get_default_args("/scratch/projects/nim00007/sam/data/mitoem")
    main(args)
