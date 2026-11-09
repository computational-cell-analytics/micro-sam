import os
import sys
import subprocess

import pytest

from micro_sam import _triton_guard

VIEWER_THEN_PLUGIN = """
import sys
import napari
viewer = napari.Viewer(show=False)
with open("/proc/self/maps") as f:
    mesa = any("libLLVM" in line for line in f)
import micro_sam.sam_annotator.annotator
import torch
print(f"mesa={mesa} triton={torch.utils._triton.has_triton_package()}")
"""


@pytest.mark.gui
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Mesa's libLLVM only clashes with triton on Linux.")
def test_plugin_import_after_viewer():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", VIEWER_THEN_PLUGIN], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    line = result.stdout.strip().splitlines()[-1]
    if line.startswith("mesa=True"):
        assert line.endswith("triton=False")


def test_block_triton_if_mesa_llvm_is_loaded(monkeypatch):
    monkeypatch.setattr(_triton_guard, "mesa_llvm_is_loaded", lambda: True)
    previous = sys.modules.pop("triton", None)
    try:
        assert _triton_guard.block_triton_if_mesa_llvm_is_loaded()
        with pytest.raises(ImportError):
            import triton  # noqa
    finally:
        sys.modules.pop("triton", None)
        if previous is not None:
            sys.modules["triton"] = previous


def test_triton_already_imported_is_kept(monkeypatch):
    monkeypatch.setattr(_triton_guard, "mesa_llvm_is_loaded", lambda: True)
    monkeypatch.setitem(sys.modules, "triton", object())
    assert not _triton_guard.block_triton_if_mesa_llvm_is_loaded()
