import json
import os
import re

import numpy as np
import pooch
import pytest

from micro_sam._provenance import (
    CONVENTION, IMAGE, image_reference, local_reference, make_step, registry_reference, timestamp,
)

# The sha256 checksum of the schema at CONVENTION["schema_url"]. Update it together with the convention version.
SCHEMA_HASH = "1d240d05cb5bf090a8a2aa57725d03d03fa8f8f3e0c8a7c40dc8c1bb6063a2c8"


def test_references(tmp_path):
    path = tmp_path / "folder" / "model.pt"
    path.parent.mkdir()
    path.write_bytes(b"weights")

    # A local file is recorded without its folder.
    reference = local_reference(path, "model", with_checksum=True)
    assert reference["id"] == "model.pt" and reference["name"] == "model"
    assert re.fullmatch("xxh128:[0-9a-f]{32}", reference["checksum"])
    assert "folder" not in json.dumps(reference)

    assert registry_reference("https://a.org/m", "xxh128:ab", "model")["checksum"] == "xxh128:ab"
    assert registry_reference("https://a.org/m", "ab", "model")["checksum"] == "sha256:ab"
    assert "checksum" not in registry_reference("https://a.org/m", None, "model")


def test_image_reference(tmp_path):
    from micro_sam._test_util import write_ome_zarr

    image, output = str(tmp_path / "image.ome.zarr"), str(tmp_path / "out.ome.zarr")
    data = np.zeros((4, 4), dtype="uint8")
    write_ome_zarr(image, data, data)

    assert image_reference(np.zeros((2, 2)), None, output) is None
    assert image_reference(None, None, output) is None
    assert image_reference(image, "s0", output) == {
        "id": "image.ome.zarr", "name": "image", "micro_sam:key": "s0", "type": IMAGE
    }
    # An input in the output file is recorded by the path of its group and the key of its array.
    assert image_reference(image, None, image) == {"path": ".", "name": "image", "micro_sam:key": "s0", "type": IMAGE}
    assert image_reference(image + os.sep, "s0", image)["path"] == "."
    expected = {"path": "labels/cells", "name": "segmentation", "micro_sam:key": "labels/cells/s0"}
    assert image_reference(image, "labels/cells/s0", image, name="segmentation", type=None) == expected
    # This also holds for a group key, and for a segmentation without a key.
    assert image_reference(image, "labels/cells", image, name="segmentation", type=None) == expected
    reference = image_reference(image, None, image, name="segmentation", type=None, is_segmentation=True)
    assert reference == expected


def test_step_matches_the_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    with open(pooch.retrieve(CONVENTION["schema_url"], known_hash=SCHEMA_HASH)) as f:
        schema = json.load(f)

    from micro_sam._test_util import write_ome_zarr

    image = str(tmp_path / "image.ome.zarr")
    write_ome_zarr(image, np.zeros((4, 4), dtype="uint8"))
    parameters = {"tile_shape": (16, 16), "threshold": np.float32(0.5), "sizes": np.arange(2), "function": len}
    step = make_step(
        "Automatic instance segmentation", "Segmentation", "micro_sam.test", parameters,
        [image_reference(image, "s0", image), None, registry_reference("https://a.org/m", "ab", "model")],
        started=timestamp(),
    )
    assert step["parameters"] == {
        "tile_shape": [16, 16], "threshold": 0.5, "sizes": [0, 1], "function": repr(len)
    }
    assert len(step["inputs"]) == 2

    attributes = json.loads(json.dumps({"zarr_conventions": [CONVENTION], "provenance:steps": [step]}))
    jsonschema.validate(attributes, schema, format_checker=jsonschema.FormatChecker())
    jsonschema.validate({"zarr_format": 3, "node_type": "group", "attributes": attributes}, schema)
