from distutils.core import setup

setup(
    name="micro_sam",
    version="0.0.1",
    description="SegmentAnything For Microscopy",
    author="Anwai Archit, Constantin Pape",
    author_email="anwai.archit@uni-goettingen.de, constantin.pape@informatik.uni-goettingen.de",
    url="https://user.informatik.uni-goettingen.de/~pape41/",
    packages=["micro_sam"],
    license="MIT",  # TODO add the license and check that it's fine with SegmentAnything
    # TODO add entry points for the napari annotator scripts
    entry_points={
        "console_scripts": [
            "micro_sam.precompute_embeddings = micro_sam.util:main",
        ]
    }
)
