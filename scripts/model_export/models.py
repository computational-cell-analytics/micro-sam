import json
import os

import numpy as np
import requests
import yaml

ADDJECTIVE_URL = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/main/adjectives.txt"
ANIMAL_URL = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/main/animals.yaml"
COLLECTION_URL = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/collection.json"

MODEL_TO_ID = {
    "vit_t_em_organelles": "greedy-whale",
    "vit_b_em_organelles": "noisy-ox",
    "vit_l_em_organelles": "humorous-crab",
    "vit_t_lm": "faithful-chicken",
    "vit_b_lm": "diplomatic-bug",
    "vit_l_lm": "idealistic-rat",
}

MODEL_TO_NAME = {
    "vit_t_em_organelles": "SAM EM Organelle Generalist (ViT-T)",
    "vit_b_em_organelles": "SAM EM Organelle Generalist (ViT-B)",
    "vit_l_em_organelles": "SAM EM Organelle Generalist (ViT-L)",
    "vit_t_lm": "SAM LM Generalist (ViT-T)",
    "vit_b_lm": "SAM LM Generalist (ViT-B)",
    "vit_l_lm": "SAM LM Generalist (ViT-L)",
}


def download_file(url, filename):
    if os.path.exists(filename):
        return

    # Send HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file in write-text mode
        with open(filename, "w", encoding=response.encoding or "utf-8") as file:
            file.write(response.text)  # Using .text instead of .content
        print(f"File '{filename}' has been downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def get_id_and_emoji(name):
    if name in MODEL_TO_ID:
        model_id = MODEL_TO_ID[name]
        adj, name = model_id.split()

    else:
        adjective_file = "adjectives.txt"
        download_file(ADDJECTIVE_URL, adjective_file)
        adjectives = []
        with open(adjective_file) as f:
            for adj in f.readlines():
                adjectives.append(adj.rstrip("\n"))

        animal_file = "animals.yaml"
        download_file(ANIMAL_URL, animal_file)
        with open(animal_file) as f:
            animal_dict = yaml.safe_load(f)
        animal_names = list(animal_dict.keys())

        collection_file = "collection.json"
        download_file(COLLECTION_URL, collection_file)
        with open(collection_file) as f:
            collection = json.load(f)["collection"]

        existing_ids = []
        for entry in collection:
            this_id = entry.get("nickname", None)
            if this_id is None:
                continue
            existing_ids.append(this_id)

        adj, name = np.random.choice(adjectives), np.random.choice(animal_names)
        model_id = f"{adj}-{name}"
        while model_id in existing_ids:
            adj, name = np.random.choice(adjectives), np.random.choice(animal_names)
            model_id = f"{adj}-{name}"

    return model_id, animal_dict[name]
