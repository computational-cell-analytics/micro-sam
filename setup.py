import runpy
from distutils.core import setup

__version__ = runpy.run_path("micro_sam/__version__.py")["__version__"]

setup(
    name="micro_sam",
    version=__version__,
    description="SegmentAnything For Microscopy",
    author="Anwai Archit, Constantin Pape",
    url="https://user.informatik.uni-goettingen.de/~pape41/",
    packages=["micro_sam"],
    license="MIT",
    entry_points={
        "console_scripts": [
            "micro_sam.annotator = micro_sam.sam_annotator.annotator:main",
            "micro_sam.annotator_2d = micro_sam.sam_annotator.annotator_2d:main",
            "micro_sam.annotator_3d = micro_sam.sam_annotator.annotator_3d:main",
            "micro_sam.annotator_tracking = micro_sam.sam_annotator.annotator_tracking:main",
            "micro_sam.image_series_annotator = micro_sam.sam_annotator.image_series_annotator:main",
            "micro_sam.precompute_embeddings = micro_sam.util:main",
        ]
    }
)
