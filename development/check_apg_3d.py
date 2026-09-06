import napari
import h5py

z = 50

data_path = "./data/N_522_final_crop_ds2.h5"
with h5py.File(data_path, "r") as f:
    data = f["raw"][z]


seg_path = "./data/seg_z50.h5"
with h5py.File(seg_path, "r") as f:
    seg = f["seg"][:]


v = napari.Viewer()
v.add_image(data)
v.add_labels(seg)
napari.run()
