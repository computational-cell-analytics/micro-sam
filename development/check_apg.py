import napari
import h5py

data_path = "./data/wsi.h5"


# FOR MASKING
# from scipy.ndimage import uniform_filter
# # Find a good masking strategy.
# with h5py.File(data_path, "r") as f:
#     image = f["data/s4"][:]
#
# print("Compute mask ...")
# bg_threshold = 240
# window = 15
# majority_threshold = 0.3
# mask = (image > bg_threshold).all(axis=-1)
# filtered = uniform_filter(mask.astype("float"), size=window)
# mask = ~(filtered >= majority_threshold)
# print("done")
#
# v = napari.Viewer()
# v.add_image(image, rgb=True)
# v.add_image(filtered)
# v.add_labels(mask.astype("uint8"), name="mask")
# napari.run()
# quit()


halo = (4000, 4000)
with h5py.File(data_path, "r") as f:
    image = f["data/s0"]
    bb = tuple(slice(3 * sh // 8 - ha, 3 * sh // 8 + ha) for sh, ha in zip(image.shape[:2], halo))
    image = image[bb]


with h5py.File("./data/seg.h5", "r") as f:
    seg = f["seg"][bb]


v = napari.Viewer()
v.add_image(image, rgb=True)
v.add_labels(seg)
napari.run()
