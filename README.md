[![DOC](https://shields.mitmproxy.org/badge/docs-pdoc.dev-brightgreen.svg)](https://computational-cell-analytics.github.io/micro-sam/)
[![PyPI](https://img.shields.io/pypi/v/micro_sam.svg)](https://pypi.org/project/micro_sam/)
[![Conda](https://anaconda.org/conda-forge/micro_sam/badges/version.svg)](https://anaconda.org/conda-forge/micro_sam)
[![codecov](https://codecov.io/gh/computational-cell-analytics/micro-sam/graph/badge.svg?token=7ETPP5CABP)](https://codecov.io/gh/computational-cell-analytics/micro-sam)
[![Image.sc forum][imagesc-img]][imagesc-url]
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7919746-blue.svg)](https://doi.org/10.5281/zenodo.7919746)

# Segment Anything for Microscopy

<a href="https://github.com/computational-cell-analytics/micro-sam"><img src="https://github.com/computational-cell-analytics/micro-sam/blob/main/doc/logo/logo_and_text.png" width="300" align="right">

Tools for segmentation and tracking in microscopy build on top of [Segment Anything](https://segment-anything.com/).
Segment and track objects in microscopy images interactively with a few clicks!

We implement napari applications for:
- interactive 2d segmentation (Left: interactive cell segmentation)
- interactive 3d segmentation (Middle: interactive mitochondria segmentation in EM)
- interactive tracking of 2d image data (Right: interactive cell tracking)

<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/d04cb158-9f5b-4460-98cd-023c4f19cccd" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/dfca3d9b-dba5-440b-b0f9-72a0683ac410" width="256">
<img src="https://github.com/computational-cell-analytics/micro-sam/assets/4263537/aefbf99f-e73a-4125-bb49-2e6592367a64" width="256">

If you run into any problems or have questions regarding our tool please open an [issue](https://github.com/computational-cell-analytics/micro-sam/issues/new/choose) on Github or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam`, and tagging [@constantinpape](https://forum.image.sc/u/constantinpape/summary) and [@anwai98](https://forum.image.sc/u/anwai98/summary).
You can follow recent updates on `micro_sam` in our [news feed](https://forum.image.sc/t/microsam-news-feed).

## Installation and Usage

Please check [the documentation](https://computational-cell-analytics.github.io/micro-sam/) for details on how to install and use `micro_sam`. You can also watch [the quickstart video](https://youtu.be/gcv0fa84mCc), [our virtual I2K workshop video](https://www.youtube.com/watch?v=dxjU4W7bCis&list=PLdA9Vgd1gxTbvxmtk9CASftUOl_XItjDN&index=33) or [all video tutorials](https://youtube.com/playlist?list=PLwYZXQJ3f36GQPpKCrSbHjGiH39X4XjSO&si=qNbB8IFXqAX33r_Z).


## Contributing

We always welcome new contributions!

If you just want to suggest a documentation edit, [you can do this directly here, on GitHub][edit-files-on-github].
For more complex changes, see [CONTRIBUTING.md][contributing] or the [contributing guide](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#contribution-guide) in our documentation for details. The first step is to [discuss your idea in a new issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) with the current developers.

Everyone participating in this project is expected to follow our [Code of Conduct][code-of-conduct].


## Citation

If you are using this repository in your research please cite
- our [paper](https://www.nature.com/articles/s41592-024-02580-4) (now published in Nature Methods!)
- and the original [Segment Anything publication](https://arxiv.org/abs/2304.02643).
- If you use `vit-tiny` models please also cite [Mobile SAM](https://arxiv.org/abs/2306.14289).
- If you use automatic tracking, please also cite [Trackastra](https://arxiv.org/abs/2405.15700).


## Related Projects

There are a few other napari plugins build around Segment Anything:
- https://github.com/MIC-DKFZ/napari-sam (2d and 3d support)
- https://github.com/royerlab/napari-segment-anything (only 2d support)
- https://github.com/hiroalchem/napari-SAM4IS

Compared to these we support more applications (2d, 3d and tracking), and provide finetuning methods and finetuned models for microscopy data.
[WebKnossos](https://webknossos.org/) and [QuPath](https://qupath.github.io/) also offer integration of Segment Anything for interactive segmentation.

We have also built follow-up work that is based on `micro_sam`:
- https://github.com/computational-cell-analytics/patho-sam - improves SAM for histopathology.
- https://github.com/computational-cell-analytics/medico-sam - improves SAM for medical imaging.
- https://github.com/computational-cell-analytics/peft-sam - studies parameter efficient fine-tuning for SAM.

## Release Overview

You can find an overview of changes introduced in previous releases [here](https://github.com/computational-cell-analytics/micro-sam/blob/main/RELEASE_OVERVIEW.md).

[imagesc-img]: https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fmicro-sam.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC
[imagesc-url]: https://forum.image.sc/tag/micro-sam
[contributing]: https://github.com/computational-cell-analytics/micro-sam/blob/main/CONTRIBUTING.md
[code-of-conduct]: https://github.com/computational-cell-analytics/micro-sam/blob/main/CODE_OF_CONDUCT.md
[edit-files-on-github]: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
