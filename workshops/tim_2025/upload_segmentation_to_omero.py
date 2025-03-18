import imageio.v3 as imageio
import numpy as np
from skimage.measure import find_contours

from omero.model import RoiI, MaskI, PolygonI
from omero.rtypes import rint, rstring
from omero_utils import omero_credential_parser, connect_to_omero


def upload_as_polygon(conn, binary_mask, image, seg_id):
    roi = RoiI()
    roi.setImage(image._obj)

    if binary_mask.ndim == 3:
        planes = binary_mask.shape[0]
    else:
        planes = 1
        binary_mask = binary_mask[np.newaxis, ...]

    for z in range(planes):
        contours = find_contours(binary_mask[z], level=0.5)
        for contour in contours:
            points = " ".join(f"{x},{y}" for y, x in contour)
            polygon = PolygonI()
            polygon.setPoints(rstring(points))
            polygon.setTheZ(rint(z))
            polygon.setTheT(rint(0))
            roi.addShape(polygon)

    roi = conn.getUpdateService().saveAndReturnObject(roi)
    return roi


def upload_as_mask(conn, binary_mask, image, seg_id):
    # Convert mask to bytes
    mask_bytes = np.packbits(binary_mask).tobytes()

    # Create Mask shape
    mask_shape = MaskI()
    mask_shape.setX(rint(0))  # X offset of mask
    mask_shape.setY(rint(0))  # Y offset of mask
    mask_shape.setWidth(rint(binary_mask.shape[1]))
    mask_shape.setHeight(rint(binary_mask.shape[0]))
    mask_shape.setBytes(mask_bytes)
    mask_shape.setTheZ(rint(0))
    mask_shape.setTheT(rint(0))
    mask_shape.setTextValue(rstring(f"Binary mask-{seg_id}"))

    # Create ROI and link to image
    roi = RoiI()
    roi.setImage(image._obj)
    roi.addShape(mask_shape)

    roi = conn.getUpdateService().saveAndReturnObject(roi)
    return roi


def upload_segmentation(conn, args):
    image = conn.getObject("Image", args.image_id)

    segmentation = imageio.imread(args.input)
    segmentation_ids = np.unique(segmentation)[1:]

    print("Uploading", len(segmentation_ids), "masks to omero")
    # TODO give some more specific metadata?
    for seg_id in segmentation_ids:
        binary_mask = (segmentation == seg_id).astype("uint8")
        if args.as_mask:
            roi = upload_as_mask(conn, binary_mask, image, seg_id)
        else:
            roi = upload_as_polygon(conn, binary_mask, image, seg_id)

        print("Uploaded the mask for id", seg_id, "to omero with ID", roi.id.val)


def main():
    parser = omero_credential_parser()
    parser.add_argument("-i", "--input", required=True)
    # If this is activated then we upload the annotations as a binary mask instead of as polygon (the defaulto)
    # Note: this does not yet work due to a possible bug in omero-py
    parser.add_argument("--as_mask", action="store_true")
    args = parser.parse_args()

    conn = connect_to_omero(args)
    upload_segmentation(conn, args)
    conn.close()


if __name__ == "__main__":
    main()
