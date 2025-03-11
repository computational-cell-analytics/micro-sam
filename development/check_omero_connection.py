"""
How to install Miniforge on HIVE?
1. PS C:\Users\architan> E:\TiM2025-Software\Miniforge3-Windows-x86_64.exe
2. Follow instructions and install the package.
3. .\miniforge3\Scripts\conda.exe init powershell (from your home directory)
"""  # noqa

import numpy as np
from shapely import LineString
from skimage.measure import find_contours

import ezomero as ez

from micro_sam.sam_annotator import annotator_2d


def _load_image(conn):
    # Step 1: Use the connecton to access files.
    # NOTE: The "project_id" info is located inside the metadata section of your project.
    # eg. the "project_id" for "example_data_test" is set to 46.
    dataset_ids = ez.get_dataset_ids(conn, project=46)

    # - Now that we know the dataset ids, the next sub-groups where data is stored,
    # we go ahead with accessing them.
    for id in dataset_ids:
        image_ids = ez.get_image_ids(conn, dataset=id)
        print(image_ids)

    # - Once we have identified the image ids, let's open just one of it.
    # I will open one z-stack.
    # NOTE: Our array is located at pixels.
    image_id = 7540
    image_obj, pixels = ez.get_image(
        conn,
        image_id=image_id,
        no_pixels=False,  # if True, only fetches the meta-data, which is super fast.
        axis_lengths=(512, 512, 40, 1, 1),  # fetches an ROI in XYZCT config, otherwise the full volume.
    )

    # Let's annotate stuff using micro-sam.
    pixels = pixels.squeeze()  # Remove singletons on-the-fly.

    # HACK: For segmentation, let's keep it simple and segment the last slice only.
    pixels = pixels[-1, ...]

    # Run the 2d annotator
    viewer = annotator_2d(image=pixels, embedding_path="test_omero.zarr", return_viewer=True)

    import napari
    napari.run()

    # Store the segmentations locally for storing them either as polygons or something else.
    segmentation = viewer.layers["committed_objects"].data

    # Let's try converting them as polygons, store them as a list of polygons and put it back.
    contours = find_contours(segmentation)[0]  # Get contours
    contour_as_line = LineString(contours)  # Convert contours to line structure.
    simple_line = contour_as_line.simplify(tolerance=1.0)  # Adjust tolerance to make the polygon.
    simple_coords = np.array(simple_line.coords)

    # Now, let's post a single ROI and see if it worked.
    ez.post_roi(conn, image_id=image_id, shapes=[ez.rois.Polygon(simple_coords)], name="test_micro_sam_seg")


def main():

    # Inspired by https://github.com/I3D-bio/omero_python_workshop.

    # Check connection to the test account.
    user = "tim2025_test"
    password = "tim2025_test"
    omero_group = "TiM2025_preparation"  # NOTE: This is always the top-level group name in hierarchy.

    host = "omero-training.gerbi-gmb.de"
    port = 4064

    # NOTE: If the default port above is blocked, use the one below.
    # host = "wss://omero-training.gerbi-gmb.de/omero-wss"
    # port = 443  # the default port (4064) seems to work for me.

    conn = ez.connect(user, password, group=omero_group, host=host, port=port, secure=True)
    print(f"Is connected: {conn.isConnected()}")

    # Visualize the image from the OMERO server.
    _load_image(conn)


if __name__ == "__main__":
    main()
