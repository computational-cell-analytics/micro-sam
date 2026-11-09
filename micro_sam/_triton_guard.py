"""Block the import of triton when Mesa's LLVM is already loaded into the process.

On Linux, the OpenGL context of napari can load the libLLVM of Mesa. Importing triton afterwards
(e.g. via torchvision -> torch._dynamo) segfaults in the static initializers of the LLVM that triton bundles.
micro_sam does not need triton, so we hide it from torch in this case.
"""

import sys


def mesa_llvm_is_loaded() -> bool:
    """Check whether a shared libLLVM is mapped into the current process.

    Returns:
        Whether libLLVM is loaded. Always False outside of Linux.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/self/maps") as f:
            return any("libLLVM" in line for line in f)
    except OSError:
        return False


def block_triton_if_mesa_llvm_is_loaded() -> bool:
    """Make 'import triton' raise an ImportError if libLLVM is loaded and triton is not imported yet.

    Returns:
        Whether triton was blocked.
    """
    if "triton" in sys.modules or not mesa_llvm_is_loaded():
        return False
    sys.modules["triton"] = None
    return True


block_triton_if_mesa_llvm_is_loaded()
