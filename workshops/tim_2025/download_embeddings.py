import os

from torch_em.data.datasets.util import download_source, unzip, download_source_gdrive


URLS_OWNCLOUD = {
    "lucchi": {
        "vit_b_em_organelles": "https://owncloud.gwdg.de/index.php/s/a2ljJVsignmItHh/download",
    },
    "nuclei_3d": {
        "vit_b": "https://owncloud.gwdg.de/index.php/s/EF9ZdMzYjDjl8fd/download",
        "vit_b_lm": "https://owncloud.gwdg.de/index.php/s/7IVekm8K7ln7yQ6/download",
    },
    "volume_em": {
        "vit_b": "https://owncloud.gwdg.de/index.php/s/1OgOEeMIK9Ok2Kj/download",
        "vit_b_em_organelles": "https://owncloud.gwdg.de/index.php/s/i9DrXe6YFL8jvgP/download",
    },
}

URLS_DRIVE = {
    "lucchi": {
        "vit_b_em_organelles": "https://drive.google.com/uc?export=download&id=1Ls1lq3eLgmiSMmPmqJdBJAmRSA57w_Ga",
    },
    "nuclei_3d": {
        "vit_b": "https://drive.google.com/uc?export=download&id=1aFkANRAqbkop2M3Df9zcZIct7Bab0jpA",
        "vit_b_lm": "https://drive.google.com/uc?export=download&id=129JvneG3th9fFXxH4iQFAFIY7_VGlupu",
    },
    "volume_em": {
        "vit_b": "https://drive.google.com/uc?export=download&id=1_4zhezz5PEX1kudPaEfxI8JfTd1AOSCd",
        "vit_b_em_organelles": "https://drive.google.com/uc?export=download&id=1K_Az5ti-P215sHvI2dCoUKHpTFX17KK8",
    },
}


CHECKSUMS = {
    "lucchi": {
        "vit_b_em_organelles": "8621591469a783c50a0fddbab1a0ff1bbfeb360f196069712960f70b1c03a9d3",
    },
    "nuclei_3d": {
        "vit_b": "82f5351486e484dda5a3a327381458515c89da5dda8a48a0b1ab96ef10d23f02",
        "vit_b_lm": "80fd701c01b81bbfb32beed6e2ece8c5706625dbc451776d8ba1c22253f097b9",
    },
    "volume_em": {
        "vit_b": "95c5e31c5e55e94780568f3fb8a3fdf33f8586a4c6a375d28dccba6567f37a47",
        "vit_b_em_organelles": "3d8d91313656fde271a48ea0a3552762f2536955a357ffb43e7c43b5b27e0627",
    },
}


def _download_embeddings(embedding_dir, dataset_name, downloader="owncloud"):
    if downloader == "drive":
        chosen_urls = URLS_DRIVE
    elif downloader == "owncloud":
        chosen_urls = URLS_OWNCLOUD
    else:
        raise ValueError(f"'{downloader}' is not a valid way to download.")

    if dataset_name is None:  # Download embeddings for all datasets.
        dataset_names = list(chosen_urls.keys())
    else:  # Download embeddings for specific dataset.
        dataset_names = [dataset_name]

    for dname in dataset_names:
        if dname not in chosen_urls:
            raise ValueError(
                f"'{dname}' does not have precomputed embeddings to download. "
                f"Please choose from {list(chosen_urls.keys())}."
            )

        urls = chosen_urls[dname]
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

            if downloader == "owncloud":
                download_source(path=zip_path, url=url, download=True, checksum=checksum)
            else:
                download_source_gdrive(path=zip_path, url=url, download=True, checksum=checksum)

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
    parser.add_argument(
        "--downloader", type=str, default="owncloud",
        help="The source of urls for downloading embeddings. The available choices are 'owncloud' or 'drive'. "
        "For downloading from drive, you need to install 'gdown' using 'conda install -c conda-forge gdown==4.6.3'."
    )
    args = parser.parse_args()

    _download_embeddings(embedding_dir=args.embedding_dir, dataset_name=args.dataset_name, downloader=args.downloader)


if __name__ == "__main__":
    main()
