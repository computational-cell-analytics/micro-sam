import os
from tqdm import tqdm
from glob import glob
from typing import Literal
from natsort import natsorted
from functools import partial

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch_em.data import datasets

from elf.io import open_file

from micro_sam.evaluation.livecell import _get_livecell_paths


DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/data"


def _process_images(
    image_paths,
    label_paths,
    split,
    base_dir,
    dataset_name,
    cell_count_criterion=None,
    limiter=None,
    ensure_connected_components=False,
    ignore_label=None,
):
    if os.path.exists(os.path.join(base_dir, split)):
        return _find_paths(base_dir, split, dataset_name)

    im_folder = os.path.join(base_dir, split, "images")
    label_folder = os.path.join(base_dir, split, "labels")
    os.makedirs(im_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    assert image_paths and len(image_paths) == len(label_paths)

    curr_image_paths, curr_label_paths = [], []
    for i, (im, label) in tqdm(
        enumerate(zip(image_paths, label_paths)), desc=f"Store '{dataset_name}' images for '{split}' split",
        total=len(image_paths) if limiter is None else limiter,
    ):
        if im.ndim == 3 and im.shape[0] == 3:  # eg. for PanNuke
            im = im.transpose(1, 2, 0)  # Make channels last for RGB images.

        if ignore_label:
            assert isinstance(ignore_label, int)
            label[label == ignore_label] = 0

        if ensure_connected_components:
            label = connected_components(label)

        cell_count = len(np.unique(label))

        # If there are no labels in ground-truth, no point in storing it
        if cell_count == 1:
            continue

        # Check for minimum cells per image.
        if cell_count < cell_count_criterion:
            continue

        # Store images in a folder.
        curr_image_path = os.path.join(im_folder, f"{dataset_name}_{i:04}.tif")
        curr_label_path = os.path.join(label_folder, f"{dataset_name}_{i:04}.tif")
        imageio.imwrite(curr_image_path, im, compression="zlib")
        imageio.imwrite(curr_label_path, label, compression="zlib")
        curr_image_paths.append(curr_image_path)
        curr_label_paths.append(curr_label_path)

        if limiter and i == limiter:  # When 'n' number of images are done, that's enough.
            break

    return curr_image_paths, curr_label_paths


def _find_paths(base_dir, split, dataset_name):
    image_paths = natsorted(glob(os.path.join(base_dir, split, "images", f"{dataset_name}_*.tif")))
    label_paths = natsorted(glob(os.path.join(base_dir, split, "labels", f"{dataset_name}_*.tif")))
    return image_paths, label_paths


def _pad_image(image, target_shape=(512, 512)):
    """NOTE: Currently applicable for PanNuke only.
    """
    pad_width = [max(0, ms - sh) for sh, ms in zip(image.shape[:2], target_shape)]
    if any(pw > 0 for pw in pad_width):
        pad_width = [(0, pad_width[0]), (0, pad_width[1])]
        if image.ndim == 3:
            pad_width += [(0, 0)]
        image = np.pad(image, pad_width)

    return image


def _norm_image(image):
    """NOTE: Currently applicable for TissueNet only.
    """
    assert image.shape[-1] == 3  # ensure channel last.
    image = image.astype("float32")
    image -= image.min(axis=(0, 1))  # NOTE: For the first two channels only.
    image /= (image.max(axis=(0, 1)) + 1e-7)  # Same as above.
    image *= 255
    image = image.astype("uint8")
    return image


def _make_center_crop(image, target_shape=(512, 512), seg=None):
    """Essential for images larger than (512, 512) for our evaluation strategy.
    """
    target_h, target_w = target_shape
    H, W = image.shape[:2]  # We assume the images coming here are YXC or YX.

    # Let's choose the smallest one if size of one axis is smaller than target
    crop_h = min(target_h, H)
    crop_w = min(target_w, W)

    def _center_crop_coords():
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        return top, left

    if seg is not None:  # Otherwise, make a smarter crop subjected to labels.
        # I'll treat non-zero as foreground.
        mask = (seg != 0)

        if np.any(mask):
            ys, xs = np.where(mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            # Center of the foreground bounding box
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2

            # Place crop centered at (cy, cx), then clip to image bounds
            top = cy - crop_h // 2
            left = cx - crop_w // 2

            top = max(0, min(top, H - crop_h))
            left = max(0, min(left, W - crop_w))
        else:   # No foreground, let's fallback to the classic center crop.
            top, left = _center_crop_coords()

    else:  # Well no segmentation? We do classic stuff anyways!
        top, left = _center_crop_coords()

    image_crop = image[top: top + crop_h, left: left + crop_w, ...]

    if seg is None:
        return image_crop
    else:
        return image_crop, seg[top: top + crop_h, left: left + crop_w]


def prepare_data_paths(dataset_name, split, base_dir):
    """This script converts all images to expected 2d images.
    """
    base_dir = os.path.join(base_dir, "benchmark_apg")
    ensure_connected_components = False
    ignore_label = None

    if dataset_name == "livecell":
        # Making center crops
        cell_count_criterion = 5
        ipaths, lpaths = _get_livecell_paths(
            input_folder=os.path.join(DATA_DIR, dataset_name), split=split,
            n_val_per_cell_type=5 if split == "val" else None,
        )

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "omnipose":
        # Making center crops for larger images
        cell_count_criterion = 3
        split = "train" if split == "val" else split  # NOTE: Since 'val' does not exist for this data.
        ipaths, lpaths = datasets.light_microscopy.omnipose.get_omnipose_paths(
            os.path.join(DATA_DIR, dataset_name), split, data_choice=["bact_phase", "worm"],
        )

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "deepbacs":
        # Make center crops for larger images.
        cell_count_criterion = 5
        image_dir, label_dir = datasets.light_microscopy.deepbacs.get_deepbacs_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, bac_type="mixed",
        )
        ipaths = natsorted(glob(os.path.join(image_dir, "*.tif")))
        lpaths = natsorted(glob(os.path.join(label_dir, "*.tif")))

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "usiigaci":
        # Make crops out of the data.
        cell_count_criterion = 3
        split = "train" if split == "test" else split
        ipaths, lpaths = datasets.light_microscopy.usiigaci.get_usiigaci_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "vicar":
        # Make crops out of it.
        cell_count_criterion = 5
        ipaths, lpaths = datasets.light_microscopy.vicar.get_vicar_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "pannuke":
        # This needs to be done as the PanNuke images are stacked together.
        cell_count_criterion = 5

        if split == "val":
            volume_path = datasets.histopathology.pannuke.get_pannuke_paths(
                path=os.path.join(DATA_DIR, "pannuke"), folds=["fold_2"], download=True,
            )[0]
        else:
            volume_path = datasets.histopathology.pannuke.get_pannuke_paths(
                path=os.path.join(DATA_DIR, "pannuke"), folds=["fold_3"], download=True,
            )[0]

        f = open_file(volume_path)
        images, labels = f["images"][:], f["labels/instances"][:]
        images = images.transpose(1, 2, 3, 0)

        # Let's pad the image and labels to match (512, 512)
        images = [_pad_image(im) for im in images]
        labels = [_pad_image(lab) for lab in labels]

        if split == "val":  # NOTE: Limiting the number of images for validation.
            images, labels = images[:200], labels[:200]

    elif dataset_name == "tissuenet":
        # This needs to be done as these are zarr images.
        cell_count_criterion = 10

        fpaths = datasets.light_microscopy.tissuenet.get_tissuenet_paths(
            path=os.path.join(DATA_DIR, "tissuenet"), split=split,
        )
        fpaths = natsorted(fpaths)
        images = [open_file(p)["raw/rgb"][:].transpose(1, 2, 0) for p in fpaths]
        labels = [open_file(p)["labels/cell"][:] for p in fpaths]

        # Let's normalize the selected channels for tissuenet
        images = [_norm_image(im) for im in images]

    elif dataset_name == "deepseas":
        # Additional work needs to be done as the labels are binary
        cell_count_criterion = 10
        ensure_connected_components = True

        ipaths, lpaths = datasets.light_microscopy.deepseas.get_deepseas_paths(
            os.path.join(DATA_DIR, dataset_name),
            split="train" if split == "val" else split,  # NOTE: Since 'val' does not exist for this data.
            download=True,
        )
        images = [imageio.imread(p) for p in ipaths]
        labels = [imageio.imread(p) for p in lpaths]

    elif dataset_name == "toiam":
        # Making splits on the fly and have a criterion for minimum number of cells.
        cell_count_criterion = 5
        ipaths, lpaths = datasets.light_microscopy.toiam.get_toiam_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
        images = [imageio.imread(p) for p in ipaths]
        labels = [imageio.imread(p) for p in lpaths]

        # We make a super simple heuristic for it. The first 1500 images for val, last 2500 for test.
        if split == "val":
            images, labels = images[:1500], labels[:1500]
        else:
            images, labels = images[1500:], labels[1500:]

    elif dataset_name == "monuseg":
        # Making splits on the fly.
        cell_count_criterion = 5
        ipaths, lpaths = datasets.histopathology.monuseg.get_monuseg_paths(
            os.path.join(DATA_DIR, dataset_name), split="train" if split == "val" else split, download=True,
        )

        images = [imageio.imread(p) for p in ipaths]
        labels = [imageio.imread(p) for p in lpaths]

        if split == "val":  # Make the validation pool small.
            images, labels = images[:20], labels[:20]

        # Finally, let's crop the images
        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "tnbc":
        # Converting container data formats
        cell_count_criterion = 10
        fpaths = datasets.histopathology.tnbc.get_tnbc_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )

        images = [open_file(p)["raw"][:].transpose(1, 2, 0) for p in fpaths]
        labels = [open_file(p)["labels/instances"][:] for p in fpaths]

    elif dataset_name == "nuinsseg":
        # Convert some 4-channel images to 3-channel (not sure why there exist though?)
        cell_count_criterion = 10
        ipaths, lpaths = datasets.histopathology.nuinsseg.get_nuinsseg_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
        if split == "val":
            ipaths, lpaths = ipaths[:50], lpaths[:50]
        else:
            ipaths, lpaths = ipaths[50:], lpaths[50:]

        images = [imageio.imread(p) for p in ipaths]
        labels = [imageio.imread(p) for p in lpaths]

        # Ensure the shape of each image
        paired = [
            (im, lab) for im, lab in zip(images, labels) if im.ndim == 3 and im.shape[-1] == 3
        ]
        images, labels = zip(*paired)
        images, labels = list(images), list(labels)

    elif dataset_name == "puma":
        # Convert container ff to tif files.
        cell_count_criterion = 10
        fpaths = datasets.histopathology.puma.get_puma_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )

        images = [open_file(p)["raw"][:].transpose(1, 2, 0) for p in fpaths]
        labels = [open_file(p)["labels/instances/nuclei"][:] for p in fpaths]

        # Let's crop the images
        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "cytodark0":
        # Convert container ff to tif files.
        cell_count_criterion = 5
        fpaths = datasets.histopathology.cytodark0.get_cytodark0_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, magnification="20x", download=True,
        )

        images = [open_file(p)["raw"][:].transpose(1, 2, 0) for p in fpaths]
        labels = [open_file(p)["labels/instances"][:] for p in fpaths]

        # Let's crop the images
        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "u20s":
        # Make crops out of large images.
        cell_count_criterion = 5
        ipaths, lpaths = datasets.light_microscopy.u20s.get_u20s_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
        if split == "val":
            ipaths, lpaths = ipaths[:25], lpaths[:25]
        else:
            ipaths, lpaths = ipaths[25:], lpaths[25:]

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "arvidsson":
        # Make crops out of large images.
        cell_count_criterion = 5
        ipaths, lpaths = datasets.light_microscopy.arvidsson.get_arvidsson_paths(
            os.path.join(DATA_DIR, dataset_name), split, download=True,
        )

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "ifnuclei":
        # Make crops out of large images.
        cell_count_criterion = 5
        ipaths, lpaths = datasets.light_microscopy.ifnuclei.get_ifnuclei_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
        if split == "val":
            ipaths, lpaths = ipaths[:10], lpaths[:10]
        else:
            ipaths, lpaths = ipaths[10:], lpaths[10:]

        images = [_make_center_crop(imageio.imread(p)) for p in ipaths]
        labels = [_make_center_crop(imageio.imread(p)) for p in lpaths]

    elif dataset_name == "dynamicnuclearnet":
        # Converting from container ff to tif files.
        cell_count_criterion = 5
        fpaths = datasets.light_microscopy.dynamicnuclearnet.get_dynamicnuclearnet_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )

        images = [open_file(p)["raw"][:] for p in fpaths]
        labels = [open_file(p)["labels"][:] for p in fpaths]

    elif dataset_name == "blastospim":
        # Converting container ff to tif files and making splits.
        cell_count_criterion = 6
        fpaths = datasets.light_microscopy.blastospim.get_blastospim_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )

        # NOTE: There's a lot of volumes here too. Just choose a few out of all.
        if split == "val":
            fpaths = fpaths[:25]  # First 25 volumes.
        else:
            fpaths = fpaths[-50:]  # Last 50 volumes.

        images = [open_file(p)["raw"][:] for p in fpaths]
        labels = [open_file(p)["labels"][:] for p in fpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

        # Next, we do smarter cropping, subjected to where objects are.
        paired = [
            _make_center_crop(im, target_shape=(512, 512), seg=lab) for im, lab in zip(images, labels)
        ]
        images, labels = zip(*paired)
        images, labels = list(images), list(labels)

    elif dataset_name == "gonuclear":
        # Converting container ff to tif files and make splits.
        cell_count_criterion = 20
        fpaths = datasets.light_microscopy.gonuclear.get_gonuclear_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )

        images = [open_file(p)["raw/nuclei"][:] for p in fpaths]
        labels = [open_file(p)["labels/nuclei"][:] for p in fpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

        if split == "val":
            images, labels = images[:250], labels[:250]
        else:
            images, labels = images[250:], labels[250:]

        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "nis3d":
        # Converting 3d LSM images to 2d and making data splits.
        cell_count_criterion = 20
        ipaths, lpaths = datasets.light_microscopy.nis3d.get_nis3d_paths(
            os.path.join(DATA_DIR, dataset_name), split="train" if split == "val" else split,
            split_type="cross-image", download=True,
        )

        images = [imageio.imread(p) for p in ipaths]
        labels = [imageio.imread(p) for p in lpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

        # Next, we do smarter cropping, subjected to where objects are.
        paired = [
            _make_center_crop(im, target_shape=(512, 512), seg=lab) for im, lab in zip(images, labels)
        ]
        images, labels = zip(*paired)
        images, labels = list(images), list(labels)

    elif dataset_name == "parhyale_regen":
        # Convert container ff stored 3d images to 2d and make splits.
        cell_count_criterion = 10
        fpaths = datasets.light_microscopy.parhyale_regen.get_parhyale_regen_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )

        images = [open_file(p)["raw"][:] for p in fpaths]
        labels = [open_file(p)["labels"][:] for p in fpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

        if split == "val":
            images, labels = images[:40], labels[:40]
        else:
            images, labels = images[40:], labels[40:]

    elif dataset_name == "covid_if":
        # Convert from container ff and make splits.
        cell_count_criterion = 10
        fpaths = datasets.light_microscopy.covid_if.get_covid_if_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )

        icells = [open_file(p)["raw/serum_IgG/s0"][:] for p in fpaths]
        inuclei = [open_file(p)["raw/nuclei/s0"][:] for p in fpaths]
        images = [
            np.stack([icell, inuc, np.zeros_like(icell)], axis=-1) for icell, inuc in zip(icells, inuclei)
        ]
        labels = [open_file(p)["labels/cells/s0"][:] for p in fpaths]

        if split == "val":
            images, labels = images[:10], labels[:10]
        else:
            images, labels = images[10:], labels[10:]

        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "hpa":
        # Make PEFT-SAM style resizing.
        cell_count_criterion = 3
        fpaths = datasets.light_microscopy.hpa.get_hpa_segmentation_paths(
            os.path.join(DATA_DIR, dataset_name), split="train" if split == "test" else split, download=False,
        )

        # Choosing the 'protein' channel only.
        images = [open_file(p)["raw/protein"][:] for p in fpaths]
        labels = [open_file(p)["labels"][:] for p in fpaths]

        for im in images:  # Check whether all images are squares.
            assert im.ndim == 2 and im.shape[0] == im.shape[1], im.shape

        # Resize the images
        from skimage.transform import resize
        raw_trafo = partial(resize, output_shape=(512, 512), anti_aliasing=True, preserve_range=True)
        label_trafo = partial(resize, output_shape=(512, 512), anti_aliasing=False, preserve_range=True)
        images = [raw_trafo(im).astype(im.dtype) for im in images]
        labels = [label_trafo(lab).astype(lab.dtype) for lab in labels]

    elif dataset_name == "mouse_embryo":
        # Convert container ff and make splits
        cell_count_criterion = 10
        fpaths = datasets.light_microscopy.mouse_embryo.get_mouse_embryo_paths(
            os.path.join(DATA_DIR, dataset_name), name="membrane",
            split="train" if split == "test" else split, download=True,
        )

        images = [open_file(p)["raw"][:] for p in fpaths]
        labels = [open_file(p)["label"][:] for p in fpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

        if split == "val":
            images, labels = images[:150], labels[:150]
        else:
            images, labels = images[150:], labels[150:]

        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "plantseg_root":
        # Convert from container ff and make splits.
        cell_count_criterion = 5
        ignore_label = 1
        fpaths = datasets.light_microscopy.plantseg.get_plantseg_paths(
            os.path.join(DATA_DIR, "plantseg"), name="root", split=split, download=True,
        )

        images = [open_file(p)["raw"][:] for p in fpaths]
        labels = [open_file(p)["label"][:] for p in fpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

        if split == "val":
            images, labels = images[:400], labels[:400]
        else:
            images, labels = images[400:], labels[400:]

        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "plantseg_ovules":
        # Convert container ff and make splits
        cell_count_criterion = 25
        fpaths = datasets.light_microscopy.plantseg.get_plantseg_paths(
            os.path.join(DATA_DIR, "plantseg"), name="ovules", split=split, download=True,
        )

        images = [open_file(p)["raw"][:] for p in fpaths]
        labels = [open_file(p)["label_with_ignore"][:] for p in fpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

        if split == "val":
            images, labels = images[:600], labels[:600]
        else:
            images, labels = images[600:], labels[600:]

        images = [_make_center_crop(im) for im in images]
        labels = [_make_center_crop(lab) for lab in labels]

    elif dataset_name == "pnas_arabidopsis":
        # Convert container ff and make splits.
        cell_count_criterion = 50
        ignore_label = 1
        fpaths = datasets.light_microscopy.pnas_arabidopsis.get_pnas_arabidopsis_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )

        if split == "val":
            fpaths = fpaths[:5]
        else:
            fpaths = fpaths[-20:]

        images = [open_file(p)["raw"][:] for p in fpaths]
        labels = [open_file(p)["labels"][:] for p in fpaths]

        # Let's slice it up
        images = [im for vol in images for im in vol]
        labels = [lab for vol in labels for lab in vol]

    else:
        raise ValueError

    image_paths, label_paths = _process_images(
        image_paths=images,
        label_paths=labels,
        split=split,
        base_dir=base_dir,
        dataset_name=dataset_name,
        limiter=100 if split == "val" else None,
        cell_count_criterion=cell_count_criterion,
        ensure_connected_components=ensure_connected_components,
        ignore_label=ignore_label,
    )

    return image_paths, label_paths


def get_image_label_paths(dataset_name: str, split: Literal["val", "test"]):
    """Returns the available / prepared 2d image and corresponding labels for APG benchmarking.
    """
    assert split in ["val", "test"]

    # Label-free
    # TODO: Add EVICAN?
    if dataset_name == "bac_mother":
        image_paths, label_paths = datasets.light_microscopy.bac_mother.get_bac_mother_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )
    elif dataset_name in ["livecell", "omnipose", "deepbacs", "usiigaci", "vicar", "deepseas", "toiam"]:
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )

    # Histopathology
    # TODO: Need one more out-of-domain data? (NuClick was lymphocytes only)
    elif dataset_name == "ihc_tma":
        image_paths, label_paths = datasets.histopathology.srsanet.get_srsanet_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )
    elif dataset_name == "lynsec":
        image_paths, label_paths = datasets.histopathology.lynsec.get_lynsec_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, choice="ihc", download=True,
        )
    elif dataset_name in ["pannuke", "monuseg", "tnbc", "nuinsseg", "puma", "cytodark0"]:
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )

    # Fluoroscence (Nuclei)
    # TODO: Another dataset? (the labels for AISegCell are horrible, the evaluation is thrown off)
    elif dataset_name == "dsb":
        image_paths, label_paths = datasets.light_microscopy.dsb.get_dsb_paths(
            os.path.join(DATA_DIR, dataset_name),
            source="reduced",
            split="train" if split == "val" else split,  # NOTE: Since 'val' does not exist for this data.
        )
    elif dataset_name == "bitdepth_nucseg":
        image_paths, label_paths = [], []
        for mag in ['20x', '40x_air', '40x_oil', '63x_oil']:
            ipaths, lpaths = datasets.light_microscopy.bitdepth_nucseg.get_bitdepth_nucseg_paths(
                os.path.join(DATA_DIR, dataset_name), magnification=mag, download=True,
            )
            if split == "val":
                image_paths.extend(ipaths[:4]), label_paths.extend(lpaths[:4])
            else:
                image_paths.extend(ipaths[4:]), label_paths.extend(lpaths[4:])
    elif dataset_name in [
        "dynamicnuclearnet", "u20s", "arvidsson", "ifnuclei",
        "blastospim", "gonuclear", "nis3d", "parhyale_regen"
    ]:
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )

    # Fluorescence (Cells)
    # TODO: Look for another data? (MouseEmbryo has really bad labels)
    elif dataset_name == "cellpose":
        image_paths, label_paths = datasets.light_microscopy.cellpose.get_cellpose_paths(
            os.path.join(DATA_DIR, dataset_name), split="train" if split == "val" else split,
            choice="cyto", download=True,
        )
    elif dataset_name == "cellbindb":
        image_paths, label_paths = [], []
        for choice in ["10×Genomics_DAPI", "DAPI", "mIF"]:
            ipaths, lpaths = datasets.light_microscopy.cellbindb.get_cellbindb_paths(
                os.path.join(DATA_DIR, dataset_name), data_choice=choice, download=True,
            )
            if split == "val":
                image_paths.extend(ipaths[:20]), label_paths.extend(lpaths[:20])
            else:
                image_paths.extend(ipaths[20:]), label_paths.extend(lpaths[20:])
    elif dataset_name in [
        "tissuenet", "plantseg_root", "covid_if", "hpa", "mouse_embryo", "plantseg_ovules", "pnas_arabidopsis"
    ]:
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )

    else:
        raise ValueError(dataset_name)

    return image_paths, label_paths
