import os
import time

import github

from models import MODEL_TO_ID


def trigger_workflow(model_id, stage_number):
    workflow_name = "publish.yaml"
    g = github.Github(login_or_token=os.environ["GITHUB_PAT"])

    repo = g.get_repo("bioimage-io/collection")
    workflow = repo.get_workflow(workflow_name)

    ref = repo.get_branch("main")
    inputs = {"resource_id": model_id, "stage_number": stage_number}
    ok = workflow.create_dispatch(ref=ref, inputs=inputs)
    assert ok


PUBLISHED_MODELS = ["vit_b_lm"]


def main():
    stage_number = 1
    for model_name, model_id in MODEL_TO_ID.items():
        if model_name in PUBLISHED_MODELS:
            print("Model", model_name, "is already uploaded")
            continue
        trigger_workflow(model_id, stage_number)
        time.sleep(1)


if __name__ == "__main__":
    main()
