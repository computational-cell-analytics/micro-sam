import h5py

from micro_sam.sam_annotator import image_series_annotator


ROOT = "/scratch/projects/nim00007/sam/data/lucchi/lucchi_test.h5"


def _get_volume(volume_path):
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]

    return raw, labels


def segment_volume():
    ...


def segment_each_slice():
    ...


def segment_each_n_slices():
    ...


def main():
    raw, _ = _get_volume(ROOT)


if __name__ == "__main__":
    main()
