import argparse
import omero
from omero.gateway import BlitzGateway

import imageio.v3 as imageio
from elf.io import open_file


def connect_to_omero(args):
    USERNAME = args.username
    PASSWORD = args.password
    HOST = "omero.tim2025.de"
    PORT = 4064  # Default OMERO port

    conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)
    conn.connect()

    if conn.isConnected():
        print("Connected to OMERO")
    else:
        print("Failed to connect")
        exit(1)

    return conn


def upload_lucchi(conn, args):
    dataset = conn.getObject("Dataset", args.dataset_id)
    file_path = "/home/pape/.cache/micro_sam/sample_data/lucchi_pp.zip.unzip/Lucchi++/Test_In"
    with open_file(file_path, "r") as f:
        x = f["*.png"][:]
    print(x.shape)
    image = conn.createImageFromNumpySeq(
        (plane for plane in x),
        dataset=dataset,
        imageName="Mitochondria in EM",
        description="FIBSEM volume of neural tissue with mitochondria and other organalles",
    )

    print(f"Created image with ID: {image.id}")


def upload_livecell(conn, args):
    dataset = conn.getObject("Dataset", args.dataset_id)
    file_path = "/home/pape/.cache/micro_sam/sample_data/livecell-2d-image.png"
    images = [imageio.imread(file_path)]
    image = conn.createImageFromNumpySeq(
        (im for im in images),
        dataset=dataset,
        imageName="LiveCell Data",
        description="Cells imaged in phase-contrast microscopy",
    )

    print(f"Created image with ID: {image.id}")


def omero_credential_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", required=True)
    parser.add_argument("-p", "--password", required=True)
    # This is the ID of the omero test dataset.
    parser.add_argument("-d", "--dataset_id", default=25780, type=int)
    # This is the ID of the Lucchi data in omero.
    # parser.add_argument("--image_id", default=108133, type=int)
    # This is the ID of the Livecell data in omero.
    parser.add_argument("--image_id", default=108177, type=int)
    return parser


def create_dataset(conn):
    # Create a new dataset
    dataset_name = "WS_51_Test"
    dataset_description = "Dataset for testing data transfer for the workshop."

    new_dataset = omero.model.DatasetI()
    new_dataset.setName(omero.rtypes.rstring(dataset_name))
    new_dataset.setDescription(omero.rtypes.rstring(dataset_description))

    # Save dataset to the server
    update_service = conn.getUpdateService()
    new_dataset = update_service.saveAndReturnObject(new_dataset)

    print(f"Created dataset '{dataset_name}' with ID: {new_dataset.id.val}")


def main():
    parser = omero_credential_parser()
    parser.add_argument("--create_dataset", "-c", action="store_true")
    args = parser.parse_args()

    conn = connect_to_omero(args)
    if args.create_dataset:
        create_dataset(conn)
    else:
        # upload_lucchi(conn, args)
        upload_livecell(conn, args)
    conn.close()


if __name__ == "__main__":
    main()
