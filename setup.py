import runpy
from distutils.core import setup

__version__ = runpy.run_path("micro_sam/__version__.py")["__version__"]

setup(
    name="micro_sam",
    version="0.0.1",
    description="SegmentAnything For Microscopy",
    author="Anwai Archit, Constantin Pape",
    url="https://user.informatik.uni-goettingen.de/~pape41/",
    packages=["micro_sam"],
    license="MIT",
    entry_points={
        "console_scripts": [
            "micro_sam.annotator_2d = micro_sam.sam_annotator.annotator_2d:main",
            "micro_sam.annotator_3d = micro_sam.sam_annotator.annotator_3d:main",
            "micro_sam.annotator_tracking = micro_sam.sam_annotator.annotator_tracking:main",
            "micro_sam.precompute_embeddings = micro_sam.util:main",
        ]
    }
)
