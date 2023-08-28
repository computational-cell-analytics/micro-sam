"""
.. include:: ../doc/start_page.md
.. include:: ../doc/installation.md
.. include:: ../doc/annotation_tools.md
.. include:: ../doc/python_library.md
"""

__version__ = "0.1.2.post1"

from .sample_data import (
    sample_data_image_series,
    sample_data_wholeslide,
    sample_data_livecell,
    sample_data_hela_2d,
    sample_data_3d,
    sample_data_tracking,
)

__all__ = (
    "sample_data_image_series",
    "sample_data_wholeslide",
    "sample_data_livecell",
    "sample_data_hela_2d",
    "sample_data_3d",
    "sample_data_tracking",
)
