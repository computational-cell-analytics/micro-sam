import os
import time

import github
import owncloud
import yaml

URL_BASE = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/%s/staged/1/files/%s"


def trigger_workflow(model_name, url):
    workflow_name = "stage.yaml"
    g = github.Github(login_or_token=os.environ["GITHUB_PAT"])

    repo = g.get_repo("bioimage-io/collection")
    workflow = repo.get_workflow(workflow_name)

    ref = repo.get_branch("main")
    inputs = {"resource_id": model_name, "package_url": url}
    ok = workflow.create_dispatch(ref=ref, inputs=inputs)
    assert ok


def get_model_name(model_zip_path):
    yaml_path = os.path.join(model_zip_path + ".unzip", "bioimageio.yaml")
    with open(yaml_path, "r") as f:
        model_name = yaml.safe_load(f)["id"]
    return model_name


# This doesn't work ...
def upload_to_owncloud(model_zip_path):
    url = "https://owncloud.gwdg.de/index.php/s/dQzw8wWHtKcaqmE"
    oc = owncloud.Client.from_public_link(url, folder_password="models")

    # url = "https://owncloud.gwdg.de"
    # oc = owncloud.Client(url)
    # oc.login(os.environ["OC_USER"], os.environ["OC_PWD"])
    # oc.get_file("tracking-flat.mp4", "test.mp4")

    fname = f"/{os.path.basename(model_zip_path)}"

    try:
        oc.file_info(fname)
        return None
    except owncloud.HTTPResponseError as e:
        if e.status_code == 404:
            pass

    assert os.path.exists(model_zip_path)
    oc.put_file(fname, model_zip_path)
    link_info = oc.share_file_with_link(fname)

    return link_info.get_link()


def upload_model_via_oc(model_zip_path):
    model_name = get_model_name(model_zip_path)
    url = upload_to_owncloud(model_zip_path)
    if url is None:
        print("The model", model_zip_path, "is already on ownCloud, skpping the upload.")
        return
    trigger_workflow(model_name, url)


OC_MODELS = {
    "vit_t_lm": "https://owncloud.gwdg.de/index.php/s/d21I2lnOjsnjLpz/download",
    "vit_b_lm": "https://owncloud.gwdg.de/index.php/s/PIYSswEX6WfP9Tj/download",
    "vit_l_lm": "https://owncloud.gwdg.de/index.php/s/APvKGYB91UMISV5/download",
    "vit_t_em_organelles": "https://owncloud.gwdg.de/index.php/s/2hxRZ7JePPpLRQN/download",
    "vit_b_em_organelles": "https://owncloud.gwdg.de/index.php/s/1yT6w2dpI3bdzrR/download",
    "vit_l_em_organelles": "https://owncloud.gwdg.de/index.php/s/21Q8YaDj7Z8HMNN/download",
}

UPLOADED_MODELS = (
    "vit_t_lm",
)


def upload_model_manual(model_zip_path):
    url = OC_MODELS[os.path.basename(model_zip_path)]
    model_id = get_model_name(model_zip_path)
    trigger_workflow(model_id, url)


def upload_all_models():
    for model_name, url in OC_MODELS.items():
        if model_name in UPLOADED_MODELS:
            print("Model", model_name, "is already uploaded")
            continue
        modality = "lm" if model_name.endswith("lm") else "em_organelles"
        model_zip_path = os.path.join(f"./exported_models/{modality}/{model_name}")
        assert os.path.exists(model_zip_path), model_zip_path

        upload_model_manual(model_zip_path)

        model_type = model_name[:5]
        model_id = get_model_name(model_zip_path)
        print("Model", model_name, "uploaded")
        print(model_name, URL_BASE % (model_id, f"{model_type}.pt"))
        print(f"{model_name}_decoder", URL_BASE % (model_id, f"{model_type}_decoder.pt"))
        print()
        time.sleep(1)


def main():
    # upload_model_via_oc("./exported_models/lm/vit_t_lm")
    # upload_model_manual("./exported_models/lm/vit_t_lm")

    upload_all_models()


if __name__ == "__main__":
    main()
