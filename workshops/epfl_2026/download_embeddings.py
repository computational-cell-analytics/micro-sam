import os

from torch_em.data.datasets.util import download_source, unzip


URLS = {
    "lucchi": {
        "vit_b_em_organelles": "https://owncloud.gwdg.de/index.php/s/ijIuSBFme1nUFdj/download",
    },
    "nuclei_3d": {
        "vit_b_lm": "https://owncloud.gwdg.de/index.php/s/EQZ3euu6PDU2iAJ/download",
    },
    "volume_em": {
        "vit_b_em_organelles": "https://owncloud.gwdg.de/index.php/s/76lDUJGQEDok3p3/download",
    },
}


CHECKSUMS = {
    "lucchi": {
        "vit_b_em_organelles": "32ef866ca8752510b126a4b5c59b34c06146356891788af6f5f81edf8eb1385b",
    },
    "nuclei_3d": {
        "vit_b_lm": "61b6a411d13ac0472771ab80e70532433ac930fb74f913310a92eee432f09777",
    },
    "volume_em": {
        "vit_b_em_organelles": "9233b781f1a9fd9c8ad20d61e38b888642183637acada802b3aa4b1ce00112d3",
    },
}


def _download_embeddings(embedding_dir, dataset_name):
    if dataset_name is None:  # Download embeddings for all datasets.
        dataset_names = list(URLS.keys())
    else:  # Download embeddings for specific dataset.
        dataset_names = [dataset_name]

    for dname in dataset_names:
        if dname not in URLS:
            raise ValueError(
                f"'{dname}' does not have precomputed embeddings to download. "
                f"Please choose from {list(URLS.keys())}."
            )

        urls = URLS[dname]
        checksums = CHECKSUMS[dname]

        data_embedding_dir = os.path.join(embedding_dir, dname)
        os.makedirs(data_embedding_dir, exist_ok=True)

        # Download the precomputed embeddings as zipfiles and unzip the embeddings per model.
        for name, url in urls.items():
            fnames = os.listdir(data_embedding_dir)
            if name in fnames:
                continue

            checksum = checksums[name]
            zip_path = os.path.join(data_embedding_dir, "embeddings.zip")

            download_source(path=zip_path, url=url, download=True, checksum=checksum)
            unzip(zip_path=zip_path, dst=data_embedding_dir)

        print(f"The precompted embeddings for '{dname}' are downloaded at {data_embedding_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download the precomputed image embeddings necessary for interactive annotation."
    )
    parser.add_argument(
        "-e", "--embedding_dir", type=str, default="./embeddings",
        help="The filepath to the folder where the precomputed image embeddings will be downloaded. "
        "By default, the embeddings will be stored in your current working directory at './embeddings'."
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default=None,
        help="The choice of volumetric dataset for which you would like to download the embeddings. "
        "By default, it downloads all the precomputed embeddings. Optionally, you can choose to download either of the "
        "volumetric datasets: 'lucchi', 'nuclei_3d' or 'volume_em'."
    )
    args = parser.parse_args()

    _download_embeddings(embedding_dir=args.embedding_dir, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
