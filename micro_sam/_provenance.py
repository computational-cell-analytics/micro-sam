"""Build and write provenance metadata for zarr outputs.

The metadata follows the provenance zarr convention, version 0.2.0:
https://github.com/BioImageTools/zarr-convention-provenance/blob/v0.2.0/README.md

A local file is recorded by its file name and not by its path, so that no directory names are stored in the output.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .__version__ import __version__

STEPS_KEY = "provenance:steps"
CONVENTIONS_KEY = "zarr_conventions"
CONVENTION = {
    "uuid": "0d4ef942-0f74-4187-ae72-016ddf083fe1",
    "schema_url":
        "https://raw.githubusercontent.com/BioImageTools/zarr-convention-provenance/refs/tags/v0.2.0/schema.json",
    "spec_url": "https://github.com/BioImageTools/zarr-convention-provenance/blob/v0.2.0/README.md",
    "name": "provenance:",
    "description": "The processing steps that produced or modified this node",
}
SOFTWARE_URI = "https://github.com/computational-cell-analytics/micro-sam"

IMAGE_ANALYSIS = {"uri": "http://edamontology.org/operation_3443", "label": "Image analysis", "ontology": "edam"}
IMAGE_SEGMENTATION = {
    "uri": "http://purl.obolibrary.org/obo/OBI_0003918", "label": "image segmentation", "ontology": "obi"
}
IMAGE = {"uri": "http://edamontology.org/data_2968", "label": "Image", "ontology": "edam"}
SEGMENTATION_MAP = {
    "uri": "http://purl.obolibrary.org/obo/OBI_0003360", "label": "image segmentation map", "ontology": "obi"
}

# The key of the array in a container file.
KEY_FIELD = "micro_sam:key"


def timestamp() -> str:
    """Get the current local time as ISO 8601 date-time with time zone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def local_reference(
    path: os.PathLike, name: str, key: Optional[str] = None, with_checksum: bool = False,
    type: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Get the data reference for a local file.

    Args:
        path: The path of the file.
        name: The role of the file, for example 'image' or 'model'.
        key: The key of the array in a container file.
        with_checksum: Whether to compute the checksum of the file.
        type: The ontology term for the kind of data.

    Returns:
        The data reference.
    """
    reference = {"id": os.path.basename(os.path.normpath(str(path))), "name": name}
    if key is not None:
        reference[KEY_FIELD] = key
    if type is not None:
        reference["type"] = type
    if with_checksum:
        from .util import _compute_hash
        reference["checksum"] = _compute_hash(path)
    return reference


def registry_reference(url: str, known_hash: Optional[str], name: str) -> Dict[str, Any]:
    """Get the data reference for a file from a model registry.

    Args:
        url: The download URL of the file.
        known_hash: The hash of the file in the registry: '<algorithm>:<hex digest>', or a bare sha256 hex digest.
        name: The role of the file, for example 'model' or 'decoder'.

    Returns:
        The data reference.
    """
    reference = {"uri": url, "name": name}
    if known_hash is not None:
        reference["checksum"] = known_hash if ":" in known_hash else f"sha256:{known_hash}"
    return reference


def image_reference(
    input_path: Any, key: Optional[str], output_path: os.PathLike,
    name: str = "image", type: Optional[Dict[str, str]] = IMAGE,
) -> Optional[Dict[str, Any]]:
    """Get the data reference for the input data of a result.

    Args:
        input_path: The input path. Any other object, for example a numpy array, has no reference.
        key: The key of the array in a container file.
        output_path: The output path. An input in the output zarr file is recorded by its path in this file.
        name: The role of the input.
        type: The ontology term for the kind of data.

    Returns:
        The data reference, or None if the input is not a file.
    """
    if not isinstance(input_path, (str, os.PathLike)):
        return None
    if os.path.realpath(str(input_path)) != os.path.realpath(str(output_path)):
        return local_reference(input_path, name, key=key, type=type)
    # The path of the group that holds the array. The spec does not define the path of the root: we use '.'.
    group = os.path.dirname(key.strip("/")) if key else ""
    reference = {"path": group or ".", "name": name}
    if key is not None:
        reference[KEY_FIELD] = key
    if type is not None:
        reference["type"] = type
    return reference


def _to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_to_json(v) for v in (value.tolist() if isinstance(value, np.ndarray) else value)]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def make_step(
    name: str,
    category: str,
    entry_point: str,
    parameters: Dict[str, Any],
    inputs: Sequence[Optional[Dict[str, Any]]],
    started: Optional[str] = None,
    category_terms: Sequence[Dict[str, str]] = (IMAGE_ANALYSIS, IMAGE_SEGMENTATION),
) -> Dict[str, Any]:
    """Get the step object for one processing step. The step ends when this function is called.

    Args:
        name: What the step did.
        category: The kind of processing.
        entry_point: The function or command that ran.
        parameters: The parameters of the step. Values that JSON cannot store are stored as their `repr`.
        inputs: The data references of the inputs. None entries are skipped.
        started: The time when the step started, see `timestamp`.
        category_terms: The ontology terms for the kind of processing.

    Returns:
        The step object.
    """
    step = {
        "name": name,
        "category": category,
        "category_terms": list(category_terms),
        "software": {"name": "micro_sam", "version": __version__, "entry_point": entry_point, "uri": SOFTWARE_URI},
        "parameters": _to_json(parameters),
        "inputs": [reference for reference in inputs if reference is not None],
    }
    if started is not None:
        step["started"] = started
    step["ended"] = timestamp()
    return step


def write_provenance(group, steps: Optional[List[Dict[str, Any]]]) -> None:
    """Set the provenance steps of a zarr node. This replaces the steps that the node has.

    Args:
        group: The zarr group or array, opened with z5py.
        steps: The step objects, in the order they ran. None or an empty list removes the steps.
    """
    attrs = group.attrs
    conventions = [
        convention for convention in (attrs[CONVENTIONS_KEY] if CONVENTIONS_KEY in attrs else [])
        if convention.get("uuid") != CONVENTION["uuid"]
    ]
    if steps:
        attrs[STEPS_KEY] = steps
        attrs[CONVENTIONS_KEY] = conventions + [CONVENTION]
        return
    if STEPS_KEY in attrs:
        del attrs[STEPS_KEY]
    if conventions:
        attrs[CONVENTIONS_KEY] = conventions
    elif CONVENTIONS_KEY in attrs:
        del attrs[CONVENTIONS_KEY]
