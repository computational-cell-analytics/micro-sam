"""Import checks for the dependency-light inference boundary."""

from configparser import ConfigParser
from pathlib import Path
import subprocess
import sys
import textwrap


def test_application_dependencies_are_optional() -> None:
    """The base installation stays free of application-only dependencies."""
    config = ConfigParser()
    config.read(Path(__file__).parents[1] / "setup.cfg")
    base = config["options"]["install_requires"].lower()
    full = config["options.extras_require"]["full"].lower()
    optional = (
        "bioimageio.core",
        "h5py",
        "imagecodecs",
        "imageio",
        "joblib",
        "kornia",
        "magicgui",
        "matplotlib",
        "napari",
        "natsort",
        "networkx",
        "pandas",
        "pyqt6",
        "python-elf",
        "scikit-learn",
        "superqt",
        "torch-em",
        "trackastra",
        "xarray",
    )

    assert all(dependency not in base for dependency in optional)
    assert all(dependency in full for dependency in optional)


def test_inference_imports_without_optional_application_packages() -> None:
    """Core model and prompt helpers import when application packages are absent."""
    script = textwrap.dedent(
        """
        import builtins

        unavailable = (
            "bioimageio",
            "elf",
            "imageio",
            "magicgui",
            "napari",
            "PyQt5",
            "PyQt6",
            "PySide6",
            "qtpy",
            "superqt",
            "torch_em",
            "trackastra",
        )
        original_import = builtins.__import__

        def headless_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == package or name.startswith(f"{package}.")
                   for package in unavailable):
                raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = headless_import

        from micro_sam.v1.prompt_based_segmentation import (
            segment_from_mask,
            segment_from_points,
        )
        from micro_sam.v1.util import (
            get_sam_model,
            precompute_image_embeddings,
            set_precomputed,
        )

        assert all(callable(function) for function in (
            get_sam_model,
            precompute_image_embeddings,
            segment_from_mask,
            segment_from_points,
            set_precomputed,
        ))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
