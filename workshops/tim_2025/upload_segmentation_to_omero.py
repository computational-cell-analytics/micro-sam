import imageio.v3 as imageio
import numpy as np

from omero.model import RoiI, MaskI
from omero.rtypes import rint, rstring
from omero_utils import omero_credential_parser, connect_to_omero


def upload_segmentation(conn, args):
    image = conn.getObject("Image", args.image_id)

    segmentation = imageio.imread(args.input)
    segmentation_ids = np.unique(segmentation)[1:]

    # TODO give some more specific metadata?
    for seg_id in segmentation_ids:
        binary_mask = (segmentation == seg_id).astype("uint8")

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
        # TODO give it some other name
        mask_shape.setTextValue(rstring('Binary mask'))

        # Create ROI and link to image
        roi = RoiI()
        roi.setImage(image._obj)
        roi.addShape(mask_shape)

        roi = conn.getUpdateService().saveAndReturnObject(roi)


def main():
    parser = omero_credential_parser()
    parser.add_argument("-i", "--input", required=True)
    # TODO set the image id of our sample lucchi data
    parser.add_argument("--image_id", default="", type=int)
    args = parser.parse_args()

    conn = connect_to_omero(args)
    upload_segmentation(conn, args)
    conn.close()


if __name__ == "__main__":
    main()
