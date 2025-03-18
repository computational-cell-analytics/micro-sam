import argparse
from omero.gateway import BlitzGateway

from elf.io import open_file


def get_dataset(conn, dataset_name):
    dataset = [ds for ds in conn.getObjects("Dataset") if ds.getName() == dataset_name]
    assert len(dataset) == 1, f"{dataset_name}: {len(dataset)}"
    dataset = dataset[0]
    return dataset


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


def upload_lucchi(conn, dataset_name):
    dataset = get_dataset(conn, dataset_name)
    file_path = "/home/pape/.cache/micro_sam/sample_data/lucchi_pp.zip.unzip/Lucchi++/Test_In"
    with open_file(file_path, "r") as f:
        x = f["*.png"][:]
    print(x.shape)
    conn.createImageFromNumpySeq(
        (plane for plane in x),
        dataset=dataset,
        imageName="Mitochondria in EM",
        description="FIBSEM volume of neural tissue with mitochondria and other organalles",
    )


def omero_credential_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", required=True)
    parser.add_argument("-p", "--password", required=True)
    parser.add_argument("-d", "--dataset", default="WS_51")
    return parser


def main():
    parser = omero_credential_parser()
    args = parser.parse_args()

    conn = connect_to_omero(args)
    dataset_name = args.dataset
    upload_lucchi(conn, dataset_name)
    conn.close()


if __name__ == "__main__":
    main()
