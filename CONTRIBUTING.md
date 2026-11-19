# Contributing to micro-sam

We welcome new contributions! This page is a short overview to get you started.
The full contribution guide, which also covers documentation builds and performance profiling, lives in [doc/contributing.md](doc/contributing.md) and is rendered as part of [our documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#contribution-guide).

Everyone participating in this project is expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions and bug reports

If you run into a problem or have a question about using micro-sam, please [open an issue](https://github.com/computational-cell-analytics/micro-sam/issues/new/choose) or reach out via [image.sc](https://forum.image.sc/) using the tag `micro-sam`.
You do not need to write code to help us: good bug reports and documentation improvements are valuable contributions.

## Discuss your ideas first

Before you start working on a larger change, please open a [new issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) to discuss your idea.
This lets you ask questions, and lets the current developers suggest the best way to implement it before you invest time in it.

## Set up a development environment

We use [git](https://git-scm.com/) for version control and [conda](https://docs.conda.io/en/latest/) to manage environments.
Clone the repository and check out the development branch:
```bash
$ git clone https://github.com/computational-cell-analytics/micro-sam.git
$ cd micro-sam
$ git checkout dev
```

Then create the environment, install the user and developer dependencies, and install micro-sam as an editable installation:
```bash
$ conda env create -f environment.yaml
$ conda activate sam2
$ python -m pip install -r requirements-dev.txt
$ python -m pip install -e .
```

## Make your changes

Changes are made branching off from the development branch:
```bash
$ git checkout dev
$ git checkout -b my-new-feature
```

We use [google style python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for all new code.
The [Python library documentation](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-the-python-library) is a good starting point for understanding how the micro-sam code is organized.
The SAM2 code is in `micro_sam/v2`, the SAM1 code of the previous version is in `micro_sam/v1` and the napari tools are in `micro_sam/sam_annotator`.

We use [flake8](https://flake8.pycqa.org/) to check the code style. Run it before you open a pull request:
```bash
$ flake8 --max-line-length=120 --per-file-ignores="__init__.py:F401" micro_sam/
```

## Run the tests

The tests are run with [pytest](https://docs.pytest.org/):
```bash
$ pytest
```

The full test suite takes a long time and downloads several models. While you develop, run only the tests that are relevant for your change, for example `pytest test/test_v2_prompt_based_segmentation.py`.
You can skip the SAM1 tests with `pytest -m "not v1"` and the napari tests with `pytest -m "not gui"`.

New code needs tests to go with it. We prefer small unit tests over integration tests; if your code is hard to unit test, it usually needs to be broken into smaller functions.
See the [full guide](doc/contributing.md#testing) for details on the test markers, on testing napari-based code and on code coverage.

## Open a pull request

Once your changes are ready, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) against the `dev` branch, not `main`.
You can [mark it as a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests) while you are still working on it and still discuss the best approach with the maintainers.

Please describe what your pull request changes and why, and link the issue it addresses.
The continuous integration runs the tests for your pull request; a maintainer will review it and may ask for changes before merging.
