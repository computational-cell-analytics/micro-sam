# Contribution Guide

* [Discuss your ideas](#discuss-your-ideas)
* [Clone the repository](#clone-the-repository)
* [Create your development environment](#create-your-development-environment)
* [Make your changes](#make-your-changes)
* [Check the code style](#check-the-code-style)
* [Testing](#testing)
    * [Run the tests](#run-the-tests)
    * [Writing your own tests](#writing-your-own-tests)
* [Open a pull request](#open-a-pull-request)
* [Optional: Build the documentation](#optional-build-the-documentation)
* [Optional: Benchmark performance](#optional-benchmark-performance)
    * [Run the benchmark script](#run-the-benchmark-script)
    * [Line profiling](#line-profiling)
    * [Snakeviz visualization](#snakeviz-visualization)
    * [Memory profiling with memray](#memory-profiling-with-memray)
* [Creating a new release](#creating-a-new-release)

Everyone who takes part in this project must follow our [Code of Conduct](https://github.com/computational-cell-analytics/micro-sam/blob/main/CODE_OF_CONDUCT.md).

## Discuss your ideas

We welcome new contributions! First, discuss your idea by opening a [new issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) in micro-sam.
This allows you to ask questions, and have the current developers make suggestions about the best way to implement your ideas.

## Clone the repository

We use [git](https://git-scm.com/) for version control.

Clone the repository, and check out the development branch:
```bash
$ git clone https://github.com/computational-cell-analytics/micro-sam.git
$ cd micro-sam
$ git checkout dev
```

## Create your development environment

We use [conda](https://docs.conda.io/en/latest/) to [manage our environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you don't have this already, install [miniforge](https://conda-forge.org/download/) to get started. You can also use [mamba](https://mamba.readthedocs.io/en/latest/) or micromamba instead of conda in the commands below.

Create the environment, install the developer dependencies, and install micro-sam as an editable installation:
```bash
$ conda env create -f environment.yaml
$ conda activate sam2
$ python -m pip install -r requirements-dev.txt
$ python -m pip install --no-deps -e .
```
The environment already contains all dependencies. `--no-deps` stops pip from installing the PyPI `sam2` on top of conda-forge's `sam-2`.

## Make your changes

Make your changes on a new branch that starts from the development branch:
```bash
$ git checkout dev
$ git checkout -b my-new-feature
```

We use [google style python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to create documentation for all new code.

You may also find it helpful to look at the [Python library documentation](#using-the-python-library), which explains the organization of the micro-sam code.
The SAM2 code is in `micro_sam/v2`, the SAM1 code of the previous version is in `micro_sam/v1` and the napari tools are in `micro_sam/sam_annotator`.

## Check the code style

We use [flake8](https://flake8.pycqa.org/) to check the code style, with a maximum line length of 120 characters.
Run it before you open a pull request:
```bash
$ flake8 --max-line-length=120 --per-file-ignores="__init__.py:F401" micro_sam/
```

## Testing

### Run the tests

The tests for micro-sam are run with [pytest](https://docs.pytest.org/en/stable/).

To run all tests:
```bash
$ pytest
```

The full test suite takes a long time and downloads several models. While you develop, run only the tests that are relevant for your change, for example:
```bash
$ pytest test/test_v2_prompt_based_segmentation.py
```

The tests use these markers:
* `v1`: tests for the SAM1 models in `micro_sam/v1`. Run them with `pytest -m "v1"`, or skip them with `pytest -m "not v1"`.
* `gui`: tests that use napari. Skip them with `pytest -m "not gui"`.
* `slow`: tests that take a long time. Skip them with `pytest -m "not slow"`.

The continuous integration runs `pytest -m "v1"` and `pytest -m "not v1"` for every pull request.

### Writing your own tests

If you have written new code, you will need to write tests to go with it.

#### Unit tests

Unit tests are the preferred style of tests for user contributions. Unit tests check small, isolated parts of the code for correctness. If your code is too complicated to write unit tests easily, you may need to consider breaking it up into smaller functions that are easier to test.

#### Tests involving napari

In cases where tests *must* use the napari viewer, [these tips might be helpful](https://napari.org/stable/plugins/test_deploy.html#tips-for-testing-napari-plugins) (in particular, the `make_napari_viewer_proxy` fixture). Mark these tests with `@pytest.mark.gui`.

These kinds of tests should be used only in limited circumstances. Developers are [advised to prefer smaller unit tests, and avoid integration tests](https://napari.org/stable/plugins/test_deploy.html#prefer-smaller-unit-tests-when-possible) wherever possible.

#### Code coverage

Pytest uses the [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) plugin to automatically determine which lines of code are covered by tests.

A short summary report is printed to the terminal output whenever you run pytest. The full results are also automatically written to a file named `coverage.xml`.

The [Coverage Gutters VSCode extension](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) is useful for visualizing which parts of the code need better test coverage. PyCharm professional [has a similar feature](https://www.jetbrains.com/pycharm/guide/tips/spot-coverage-in-gutter/), and you may be able to find similar tools for your preferred editor.

We also use [codecov.io](https://app.codecov.io/gh/computational-cell-analytics/micro-sam) to display the code coverage results from our GitHub Actions continuous integration.

## Open a pull request

Once you've made changes to the code and written some tests to go with it, you are ready to [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). You can [mark your pull request as a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests) if you are still working on it, and still get the benefit of discussing the best approach with maintainers.

Open your pull request against the `dev` branch, not `main`.
Describe what your pull request changes and why, and link the issue it addresses.
The continuous integration runs the tests for your pull request. A maintainer will review it and may ask for changes before merging.

## Optional: Build the documentation

We use [pdoc](https://pdoc.dev/docs/pdoc.html) to build the documentation.

To build the documentation locally, run this command:
```bash
$ python build_doc.py
```

This will start a local server and display the HTML documentation. Any changes you make to the documentation will be updated in real time (you may need to refresh your browser to see the changes).

If you want to save the HTML files, append `--out` to the command, like this:
```bash
$ python build_doc.py --out
```

This will save the HTML files into a new directory named `tmp`.

You can add content to the documentation in two ways:
1. By adding or updating [google style python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) in the micro-sam code.
    * [pdoc](https://pdoc.dev/docs/pdoc.html) will automatically find and include docstrings in the documentation.
2. By adding or editing markdown files in the micro-sam `doc` directory.
    * If you add a new markdown file to the documentation, you must tell [pdoc](https://pdoc.dev/docs/pdoc.html) that it exists by adding a line to the `micro_sam/__init__.py` module docstring (eg: `.. include:: ../doc/my_amazing_new_docs_page.md`). Otherwise it will not be included in the final documentation build!

The documentation is built and deployed to [GitHub Pages](https://computational-cell-analytics.github.io/micro-sam/) automatically by the [`build_documentation` GitHub Action](https://github.com/computational-cell-analytics/micro-sam/blob/main/.github/workflows/build_docs.yaml); there is no need to build and deploy it manually. This workflow runs whenever a new release is created, on pushes to `main` that touch the Python or markdown documentation files, and on manual dispatch from the Actions tab.

## Optional: Benchmark performance

There are a number of options you can use to benchmark performance, and identify problems like slow run times or high memory use in micro-sam.

* [Run the benchmark script](#run-the-benchmark-script)
* [Line profiling](#line-profiling)
* [Snakeviz visualization](#snakeviz-visualization)
* [Memory profiling with memray](#memory-profiling-with-memray)

### Run the benchmark script

The script `development/benchmark.py` benchmarks the SAM1 models of the previous version (`micro_sam.v1`). It measures the image embeddings, the prompt-based segmentation and the automatic mask generation.

To run the benchmark script from the root of the repository:
```bash
$ python development/benchmark.py --model_type vit_t --device cpu
```

For more details about the user input arguments for the micro-sam benchmark script, see the help:
```bash
$ python development/benchmark.py --help
```

### Line profiling

For more detailed line by line performance results, we can use [line-profiler](https://github.com/pyutils/line_profiler). It is part of the developer dependencies in `requirements-dev.txt`.

> [line_profiler](https://github.com/pyutils/line_profiler) is a module for doing line-by-line profiling of functions. kernprof is a convenient script for running either `line_profiler` or the Python standard library's cProfile or profile modules, depending on what is available.

To do line-by-line profiling:
1. Add the `@profile` decorator to any function in the call stack.
2. Run `kernprof -lv development/benchmark.py --model_type vit_t --device cpu`.

For more details about how to use line-profiler and kernprof, see [the documentation](https://kernprof.readthedocs.io/en/latest/).

### Snakeviz visualization

For more detailed visualizations of profiling results, we use [snakeviz](https://jiffyclub.github.io/snakeviz/). It is part of the developer dependencies in `requirements-dev.txt`.

> SnakeViz is a browser based graphical viewer for the output of Python’s cProfile module.

1. Generate a profile file: `python -m cProfile -o program.prof development/benchmark.py --model_type vit_t --device cpu`
2. Visualize the profile file: `snakeviz program.prof`

For more details about how to use snakeviz, see [the documentation](https://jiffyclub.github.io/snakeviz/).

### Memory profiling with memray

If you need to investigate memory use specifically, we use [memray](https://github.com/bloomberg/memray). Install it with `python -m pip install memray`.

> Memray is a memory profiler for Python. It can track memory allocations in Python code, in native extension modules, and in the Python interpreter itself. It can generate several different types of reports to help you analyze the captured memory usage data. While commonly used as a CLI tool, it can also be used as a library to perform more fine-grained profiling tasks.

For more details about how to use memray, see [the documentation](https://bloomberg.github.io/memray/getting_started.html).

## Creating a new release

Only maintainers create releases.

To create a new release:
1. Update the version number in [micro_sam/__version__.py](https://github.com/computational-cell-analytics/micro-sam/blob/main/micro_sam/__version__.py) and add an entry for the new version to [RELEASE_OVERVIEW.md](https://github.com/computational-cell-analytics/micro-sam/blob/main/RELEASE_OVERVIEW.md).
2. Merge these changes into `main` with a pull request.

The CI then tags the new version, publishes the GitHub release and uploads the package to PyPI.

Pre-releases of a new major version, such as `2.0.0a1`, are only annotated tags on the `dev` branch. They do not have a GitHub release or a PyPI upload.
