import importlib
from pathlib import Path

import pytest
import yaml

import micro_sam


def _napari_command_python_names():
    """Read all command 'python_name' targets from the napari plugin manifest."""
    manifest = Path(micro_sam.__file__).parent / "napari.yaml"
    spec = yaml.safe_load(manifest.read_text())
    return [command["python_name"] for command in spec["contributions"]["commands"]]


@pytest.mark.parametrize("python_name", _napari_command_python_names())
def test_napari_manifest_command_importable(python_name):
    """Resolve each napari plugin command exactly as napari does when loading the plugin.

    This catches import-time failures (e.g. a numpy ABI mismatch in a compiled transitive
    dependency such as 'edt') that otherwise only surface as a cryptic
    "RuntimeError: Failed to import command at '...:Annotator2d'" when opening the plugin
    in napari. We only resolve the command target (no widget instantiation), so this does
    not require a Qt event loop and runs as a non-GUI test.
    """
    module_name, _, attr = python_name.partition(":")
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), f"could not resolve napari command target '{python_name}'"
