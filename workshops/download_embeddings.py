import os

from torch_em.data.datasets.util import download_source, unzip


URLS = {
    "lucchi": [
        "https://owncloud.gwdg.de/index.php/s/kQMA1B8L9LOvYrl/download",  # vit_b
        "https://owncloud.gwdg.de/index.php/s/U8xs6moRg0cQhkS/download",  # vit_b_em_organelles
    ],
    "embedseg": [
        "https://owncloud.gwdg.de/index.php/s/EF9ZdMzYjDjl8fd/download",  # vit_b
        "https://owncloud.gwdg.de/index.php/s/7IVekm8K7ln7yQ6/download",  # vit_b_lm
    ],
    "platynereis": [
        "https://owncloud.gwdg.de/index.php/s/1OgOEeMIK9Ok2Kj/download",  # vit_b
        "https://owncloud.gwdg.de/index.php/s/i9DrXe6YFL8jvgP/download",  # vit_b_em_organelles
    ],
}

CHECKSUMS = {
    "lucchi": [
        "e0d064765f1758a1a0823b2c02d399caa5cae0d8ac5a1e2ed96548a647717433",  # vit_b
        "e0b5ab781c42e6f68b746fc056c918d56559ccaeedb4e4f2848b1e5e8f1bec58",  # vit_b_em_organelles
    ],
    "embedseg": [
        "82f5351486e484dda5a3a327381458515c89da5dda8a48a0b1ab96ef10d23f02",  # vit_b
        "80fd701c01b81bbfb32beed6e2ece8c5706625dbc451776d8ba1c22253f097b9",  # vit_b_lm
    ],
    "platynereis": [
        "95c5e31c5e55e94780568f3fb8a3fdf33f8586a4c6a375d28dccba6567f37a47",  # vit_b
        "3d8d91313656fde271a48ea0a3552762f2536955a357ffb43e7c43b5b27e0627",  # vit_b_em_organelles
    ],
}


def _download_embeddings(embedding_dir, dataset_name):
    if dataset_name is None:  # Download embeddings for all datasets.
        dataset_names = list(URLS.keys())
    else:  # Download embeddings for specific dataset.
        dataset_names = [dataset_name]

    for dname in dataset_names:
        if dname not in URLS:
            raise ValueError(
                f"'{dname}' does not have precomputed embeddings to download. Please choose from {list(URLS.keys())}"
            )

        urls = URLS[dname]
        checksums = CHECKSUMS[dname]

        data_embedding_dir = os.path.join(embedding_dir, dname)
        os.makedirs(data_embedding_dir, exist_ok=True)

        # Download the precomputed embeddings as zipfiles and unzip the embeddings per model.
        for url, checksum in zip(urls, checksums):
            if all([p.startswith("vit_b") for p in os.listdir(data_embedding_dir)]):
                continue

            zip_path = os.path.join(data_embedding_dir, "embeddings.zip")
            download_source(path=zip_path, url=url, download=True, checksum=checksum)
            unzip(zip_path=zip_path, dst=data_embedding_dir)

        print(f"The precompted embeddings for '{dname}' are downloaded at f{data_embedding_dir}")


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
        "volumetric datasets: 'lucchi', 'embedseg' or 'platynereis'."
    )
    args = parser.parse_args()

    _download_embeddings(embedding_dir=args.embedding_dir, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
