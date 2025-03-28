import os
from typing import Union

import zarr


# TODO add a test for this (with a prepared commit file)
def rerun_segmentation_from_commit_file(
    commit_file: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
) -> None:
    """

    Args:
        commit_file:
        input_path:
    """
    f = zarr.open(commit_file, mode="r")

    # TODO
    # 1. Load the model according to the model description stored in the commit file.

    # 2. Go through the commit history and redo the action of each commit.
    # Actions can be:
    # - Committing an automatic segmentation result.
    # - Committing an interactive segmentation result.


# TODO
def continue_annotation_from_commit_file(
    commit_file: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
) -> None:
    """
    """


# TODO CLI for 'continue_annotation_from_commit_file'
def main():
    pass
