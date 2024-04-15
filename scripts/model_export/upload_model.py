import os

import github
import owncloud
import yaml


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
    "vit_t_lm": "https://owncloud.gwdg.de/index.php/s/d21I2lnOjsnjLpz/download"
}


# TODO check which models have been uploaded and skip them
# So theat we can just do this in a loop
def upload_model_manual(model_zip_path):
    url = OC_MODELS[os.path.basename(model_zip_path)]
    model_name = get_model_name(model_zip_path)
    trigger_workflow(model_name, url)


# TODO auto-generate the links
# https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/acclaimed-angelfish/staged/1/files/bioimageio.yaml
# https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/faithful-chicken/staged/1/files/bioimageio.yaml
def main():
    # upload_model_via_oc("./exported_models/lm/vit_t_lm")
    upload_model_manual("./exported_models/lm/vit_t_lm")


if __name__ == "__main__":
    main()
