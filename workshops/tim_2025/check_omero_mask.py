import warnings

import napari
import numpy as np
from omero_utils import omero_credential_parser, connect_to_omero
from skimage.draw import polygon


def load_masks(conn, segmentation, image_id, mask_ids):
    roi_service = conn.getRoiService()
    # Load all ROIs linked to the image
    result = roi_service.findByImage(image_id, None)

    for roi in result.rois:
        roi_id = roi.getId().val
        if roi_id not in mask_ids:
            continue
        print("Loading mask", roi_id)

        for shape in roi.copyShapes():
            if shape.__class__.__name__ == "MaskI":
                width = int(shape.getWidth().getValue())
                height = int(shape.getHeight().getValue())
                mask_bytes = shape.getBytes()
                mask_bits = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8))
                mask_array = (mask_bits[:width * height].reshape((height, width)) > 0)
            elif shape.__class__.__name__ == "PolygonI":
                image = conn.getObject("Image", image_id)
                height, width = int(image.getSizeY()), int(image.getSizeX())
                points_str = shape.getPoints().getValue()
                points = np.array([list(map(float, p.split(','))) for p in points_str.strip().split()])
                rr, cc = polygon(points[:, 1], points[:, 0], (height, width))
                mask_array = np.zeros((height, width), dtype=bool)
                mask_array[rr, cc] = True
            else:
                warnings.warn(
                    f"Converting {shape.__class__.__name__} to a mask is currently not supported."
                    f"The ID {roi_id} is skipped."
                )
            segmentation[mask_array > 0] = roi_id

    return segmentation


def main():
    parser = omero_credential_parser()
    parser.add_argument("--mask_ids", type=int, nargs="+", default=[1327744])
    args = parser.parse_args()

    conn = connect_to_omero(args)

    # Load the image data.
    image = conn.getObject("Image", args.image_id)
    # Load image as numpy array
    image_data = image.getPrimaryPixels().getPlane()
    image_data = image_data.squeeze()

    # Load the mask data.
    segmentation = np.zeros(image_data.shape, dtype="uint64")
    load_masks(conn, segmentation, args.image_id, args.mask_ids)

    conn.close()

    v = napari.Viewer()
    v.add_image(image_data)
    v.add_labels(segmentation)
    napari.run()


if __name__ == "__main__":
    main()
