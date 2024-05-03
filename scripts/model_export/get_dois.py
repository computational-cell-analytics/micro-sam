
import requests
from models import MODEL_TO_ID

URL_BASE = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/%s/versions.json"


def get_zenodo_url(model_name, model_id):

    url = URL_BASE % model_id
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
    else:
        print("Failed to retrieve data:", response.status_code)

    doi = data["published"]["1"]["doi"]
    doi = f"https://doi.org/{doi}"

    print(model_name, ":")
    print(model_id)
    print(doi)
    print()


for model_name, model_id in MODEL_TO_ID.items():
    get_zenodo_url(model_name, model_id)
