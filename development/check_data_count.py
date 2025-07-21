import os
from glob import glob

import numpy as np
import imageio.v3 as imageio

from torch_em.data import datasets

from elf.io import open_file


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam/data"


def check_data_count(lm_version="v3"):
    image_counter, object_counter = 0, 0

    # LIVECell data.
    image_paths, label_paths = datasets.light_microscopy.livecell.get_livecell_paths(
        path=os.path.join(ROOT, "livecell"), split="train",
    )
    image_counter += len(image_paths)
    object_counter += sum(
        [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
    )

    print("LIVECell", image_counter, object_counter)

    # DeepBacs data.
    image_dir, label_dir = datasets.light_microscopy.deepbacs.get_deepbacs_paths(
        path=os.path.join(ROOT, "deepbacs"), bac_type="mixed", split="train",
    )
    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.tif")))

    curr_image_counter = len(image_paths)
    curr_object_counter = sum(
        [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
    )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("DeepBacs", curr_image_counter, curr_object_counter)

    # TissueNet data.
    sample_paths = datasets.light_microscopy.tissuenet.get_tissuenet_paths(
        path=os.path.join(ROOT, "tissuenet"), split="train",
    )
    curr_image_counter = len(sample_paths)
    curr_object_counter = sum(
        [len(np.unique(open_file(p)["labels/cell"])[1:]) for p in sample_paths]
    )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("TissueNet", curr_image_counter, curr_object_counter)

    # PlantSeg (Root) data.
    volume_paths = datasets.light_microscopy.plantseg.get_plantseg_paths(
        path=os.path.join(ROOT, "plantseg"), name="root", split="train",
    )
    curr_image_counter, curr_object_counter = 0, 0
    for p in volume_paths:
        f = open_file(p)
        curr_image_counter += f["raw"].shape[0]
        curr_object_counter += sum(
            [len(np.unique(curr_label)[1:]) for curr_label in f["label"]]
        )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("PlantSeg (Root)", curr_image_counter, curr_object_counter)

    # NeurIPS CellSeg data.
    image_paths, label_paths = datasets.light_microscopy.neurips_cell_seg.get_neurips_cellseg_paths(
        root=os.path.join(ROOT, "neurips_cellseg"), split="train",
    )
    curr_image_counter = len(image_paths)
    curr_object_counter = sum(
        [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
    )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("TissueNet", curr_image_counter, curr_object_counter)

    # CTC data.
    curr_image_counter, curr_object_counter = 0, 0
    for dataset_name in datasets.ctc.CTC_CHECKSUMS["train"].keys():
        if dataset_name in ["Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa"]:
            continue

        image_dirs, label_dirs = datasets.light_microscopy.ctc.get_ctc_segmentation_paths(
            path=os.path.join(ROOT, "ctc"), dataset_name=dataset_name,
        )
        image_paths = [p for d in image_dirs for p in sorted(glob(os.path.join(d, "*.tif")))]
        label_paths = [p for d in label_dirs for p in sorted(glob(os.path.join(d, "*.tif")))]

        curr_image_counter += len(image_paths)
        curr_object_counter += sum(
            [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
        )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("CTC", curr_image_counter, curr_object_counter)

    # DSB Nucleus data.
    image_paths, label_paths = datasets.light_microscopy.dsb.get_dsb_paths(
        path=os.path.join(ROOT, "dsb"), source="reduced", split="train",
    )
    curr_image_counter = len(image_paths)
    curr_object_counter = sum(
        [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
    )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("DSB Nucleus", curr_image_counter, curr_object_counter)

    if lm_version == "v2":
        return image_counter, object_counter

    # EmbedSeg data.
    curr_image_counter, curr_object_counter = 0, 0
    names = [
        "Mouse-Organoid-Cells-CBG", "Mouse-Skull-Nuclei-CBG", "Platynereis-ISH-Nuclei-CBG", "Platynereis-Nuclei-CBG",
    ]
    for name in names:
        image_paths, label_paths = datasets.light_microscopy.embedseg_data.get_embedseg_paths(
            path=os.path.join(ROOT, "embedseg"), name=name, split="train",
        )
        curr_image_counter += sum(
            [imageio.imread(p).shape[0] for p in image_paths]
        )
        curr_object_counter += sum(
            [sum(len(np.unique(curr_label)[1:]) for curr_label in imageio.imread(p)) for p in label_paths]
        )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("EmbedSeg", curr_image_counter, curr_object_counter)

    # CVZ Fluo data.
    curr_image_counter, curr_object_counter = 0, 0
    for stain_choice in ["cell", "dapi"]:
        image_paths, label_paths = datasets.light_microscopy.cvz_fluo.get_cvz_fluo_paths(
            path=os.path.join(ROOT, "cvz"), stain_choice=stain_choice,
        )
        curr_image_counter += len(image_paths)
        curr_object_counter += sum(
            [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
        )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("CVZ Fluo", curr_image_counter, curr_object_counter)

    # DynamicNuclearNet data.
    sample_paths = datasets.light_microscopy.dynamicnuclearnet.get_dynamicnuclearnet_paths(
        path=os.path.join(ROOT, "dynamicnuclearnet")
    )
    curr_image_counter, curr_object_counter = 0, 0
    for p in sample_paths:
        f = open_file(p)
        breakpoint()

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("DynamicNuclearNet", curr_image_counter, curr_object_counter)

    # CellPose data.
    image_paths, label_paths = datasets.light_microscopy.cellpose.get_cellpose_paths(
        path=os.path.join(ROOT, "cellpose"), split="train", choice="cyto",
    )
    curr_image_counter = len(image_paths)
    curr_object_counter = sum(
        [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
    )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("CellPose", curr_image_counter, curr_object_counter)

    # OmniPose data.
    image_paths, label_paths = datasets.light_microscopy.omnipose.get_omnipose_paths(
        path=os.path.join(ROOT, "omnipose"), split="train",
    )
    curr_image_counter = len(image_paths)
    curr_object_counter = sum(
        [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
    )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("OmniPose", curr_image_counter, curr_object_counter)

    # OrgaSegment data.
    image_paths, label_paths = datasets.light_microscopy.orgasegment.get_orgasegment_paths(
        path=os.path.join(ROOT, "orgasegment"), split="train",
    )
    curr_image_counter = len(image_paths)
    curr_object_counter = sum(
        [len(np.unique(imageio.imread(p))[1:]) for p in label_paths]
    )

    image_counter += curr_image_counter
    object_counter += curr_object_counter

    print("OrgaSegment", curr_image_counter, curr_object_counter)

    return image_counter, object_counter


def main():
    image_counts, object_counts = check_data_count("v2")
    print(f"Count of images: '{image_counts}'; and count of objects: '{object_counts}'")

    image_counts, object_counts = check_data_count("v3")
    print(f"Count of images: '{image_counts}'; and count of objects: '{object_counts}'")


if __name__ == "__main__":
    main()
