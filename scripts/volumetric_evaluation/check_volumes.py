import os
from glob import glob

import h5py


ROOT = "/media/anwai/ANWAI/data/"


def _crop_volume(raw, labels, crop_values):
    z_val, y_val, x_val = crop_values

    z_start, z_stop = z_val
    y_start, y_stop = y_val
    x_start, x_stop = x_val

    raw = raw[
        z_start: z_stop,
        y_start: y_stop,
        x_start: x_stop
    ]
    labels = labels[
        z_start: z_stop,
        y_start: y_stop,
        x_start: x_stop
    ]

    return raw, labels


def check_plantseg_volume(volume_path, set_name, crop_values=None, view=False, save_crop=False):
    volume_name = os.path.split(volume_path)[-1]
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["label"][:]

        print(f"Processing '{volume_path}'")
        print("Original shape:", raw.shape)
        if crop_values is not None:
            raw, labels = _crop_volume(raw, labels, crop_values)
            print("Cropped shape:", raw.shape)
            print()

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(labels, visible=False)
            napari.run()

    if save_crop:
        with h5py.File(os.path.join(ROOT, "for_micro_sam", f"{set_name}_{volume_name}"), "a") as f:
            f.create_dataset("volume/cropped/raw", data=raw, compression="gzip")
            f.create_dataset("volume/cropped/labels", data=labels, compression="gzip")


def _just_view_volumes():
    all_volume_paths = glob(os.path.join(ROOT, "for_micro_sam", "*.h5"))
    for volume_path in all_volume_paths:
        with h5py.File(volume_path, "r") as f:
            raw = f["volume/cropped/raw"][:]
            labels = f["volume/cropped/labels"][:]

            import napari
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(labels)
            napari.run()


def main():
    view = False

    _just_view_volumes()
    return

    # for root - val
    #  - cropped volume shape: (50, 340, 500)
    check_plantseg_volume(
        volume_path=os.path.join(ROOT, "plantseg", "root_train", "Movie1_t00040_crop_gt.h5"),
        set_name="plantseg_root_val",
        crop_values=[(60, 110), (200, None), (100, 600)],
        view=view,
        save_crop=True
    )

    # for root - test
    #  - cropped volume shape: (200, 250, 600)
    check_plantseg_volume(
        volume_path=os.path.join(ROOT, "plantseg", "root_test", "Movie1_t00045_crop_gt.h5"),
        set_name="plantseg_root_test",
        crop_values=[(None, None), (200, None), (200, 800)],
        view=view,
        save_crop=True
    )

    # for ovules - val
    #  - cropped volume shape: (50, 680, 750)
    check_plantseg_volume(
        volume_path=os.path.join(ROOT, "plantseg", "ovules_val", "N_420_ds2x.h5"),
        set_name="plantseg_ovules_val",
        crop_values=[(100, 150), (None, -200), (500, 1250)],
        view=view,
        save_crop=True
    )

    # for ovules - test
    #  - cropped volume shape: (300, 750, 750)
    check_plantseg_volume(
        volume_path=os.path.join(ROOT, "plantseg", "ovules_test", "N_435_final_crop_ds2.h5"),
        set_name="plantseg_ovules_test",
        crop_values=[(50, 350), (351, None), (None, 750)],
        view=view,
        save_crop=True
    )


if __name__ == "__main__":
    main()
