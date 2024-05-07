"""
.. include:: ../doc/start_page.md
.. include:: ../doc/installation.md
.. include:: ../doc/annotation_tools.md
.. include:: ../doc/python_library.md
.. include:: ../doc/finetuned_models.md
.. include:: ../doc/faq.md
.. include:: ../doc/contributing.md
.. include:: ../doc/band.md
"""
import os

from .__version__ import __version__

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
