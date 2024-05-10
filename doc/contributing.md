# Contribution Guide

* [Discuss your ideas](#discuss-your-ideas)
* [Clone the repository](#clone-the-repository)
* [Create your development environment](#create-your-development-environment)
* [Make your changes](#make-your-changes)
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

### Discuss your ideas

We welcome new contributions! First, discuss your idea by opening a [new issue](https://github.com/computational-cell-analytics/micro-sam/issues/new) in micro-sam.
This allows you to ask questions, and have the current developers make suggestions about the best way to implement your ideas.

### Clone the repository

We use [git](https://git-scm.com/) for version control.

Clone the repository, and checkout the development branch:
```bash
$ git clone https://github.com/computational-cell-analytics/micro-sam.git
$ cd micro-sam
$ git checkout dev
```

### Create your development environment

We use [conda](https://docs.conda.io/en/latest/) to [manage our environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you don't have this already, install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/) to get started.

Now you can create the environment, install user and developer dependencies, and micro-sam as an editable installation:
```bash
$ mamba env create environment_gpu.yaml
$ mamba activate sam
$ python -m pip install requirements-dev.txt
$ python -m pip install -e .
```

### Make your changes

Now it's time to make your code changes.

Typically, changes are made branching off from the development branch. Checkout `dev` and then create a new branch to work on your changes.
```
$ git checkout dev
$ git checkout -b my-new-feature
```

We use [google style python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to create documentation for all new code.

You may also find it helpful to look at this [developer guide](#for-developers), which explains the organization of the micro-sam code.

## Testing

### Run the tests

The tests for micro-sam are run with [pytest](https://docs.pytest.org/en/7.4.x/)

To run the tests:
```bash
$ pytest
```

### Writing your own tests

If you have written new code, you will need to write tests to go with it.

#### Unit tests

Unit tests are the preferred style of tests for user contributions. Unit tests check small, isolated parts of the code for correctness. If your code is too complicated to write unit tests easily, you may need to consider breaking it up into smaller functions that are easier to test.

#### Tests involving napari

In cases where tests *must* use the napari viewer, [these tips might be helpful](https://napari.org/stable/plugins/test_deploy.html#tips-for-testing-napari-plugins) (in particular, the `make_napari_viewer_proxy` fixture).

These kinds of tests should be used only in limited circumstances. Developers are [advised to prefer smaller unit tests, and avoid integration tests](https://napari.org/stable/plugins/test_deploy.html#prefer-smaller-unit-tests-when-possible) wherever possible.

#### Code coverage

Pytest uses the [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) plugin to automatically determine which lines of code are covered by tests.

A short summary report is printed to the terminal output whenever you run pytest. The full results are also automatically written to a file named `coverage.xml`.

The [Coverage Gutters VSCode extension](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) is useful for visualizing which parts of the code need better test coverage. PyCharm professional [has a similar feature](https://www.jetbrains.com/pycharm/guide/tips/spot-coverage-in-gutter/), and you may be able to find similar tools for your preferred editor.

We also use [codecov.io](https://app.codecov.io/gh/computational-cell-analytics/micro-sam) to display the code coverage results from our Github Actions continuous integration.

### Open a pull request

Once you've made changes to the code and written some tests to go with it, you are ready to [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). You can [mark your pull request as a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests) if you are still working on it, and still get the benefit of discussing the best approach with maintainers.

Remember that typically changes to micro-sam are made branching off from the development branch. So, you will need to open your pull request to merge back into the `dev` branch [like this](https://github.com/computational-cell-analytics/micro-sam/compare/dev...dev).

### Optional: Build the documentation

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

### Optional: Benchmark performance

There are a number of options you can use to benchmark performance, and identify problems like slow run times or high memory use in micro-sam.

* [Run the benchmark script](#run-the-benchmark-script)
* [Line profiling](#line-profiling)
* [Snakeviz visualization](#snakeviz-visualization)
* [Memory profiling with memray](#memory-profiling-with-memray)

#### Run the benchmark script

There is a performance benchmark script available in the micro-sam repository at `development/benchmark.py`.

To run the benchmark script:
```bash
$ python development/benchmark.py --model_type vit_t --device cpu`
```

For more details about the user input arguments for the micro-sam benchmark script, see the help:
```bash
$ python development/benchmark.py --help
```

#### Line profiling

For more detailed line by line performance results, we can use [line-profiler](https://github.com/pyutils/line_profiler).

> [line_profiler](https://github.com/pyutils/line_profiler) is a module for doing line-by-line profiling of functions. kernprof is a convenient script for running either `line_profiler` or the Python standard library's cProfile or profile modules, depending on what is available.

To do line-by-line profiling:
1. Ensure you have line profiler installed: `python -m pip install line_profiler`
2. Add `@profile` decorator to any function in the call stack
3. Run `kernprof -lv benchmark.py --model_type vit_t --device cpu`

For more details about how to use line-profiler and kernprof, see [the documentation](https://kernprof.readthedocs.io/en/latest/).

For more details about the user input arguments for the micro-sam benchmark script, see the help:
```bash
$ python development/benchmark.py --help
```

#### Snakeviz visualization

For more detailed visualizations of profiling results, we use [snakeviz](https://jiffyclub.github.io/snakeviz/).

> SnakeViz is a browser based graphical viewer for the output of Python’s cProfile module.

1. Ensure you have snakeviz installed: `python -m pip install snakeviz`
2. Generate profile file: `python -m cProfile -o program.prof benchmark.py --model_type vit_h --device cpu`
3. Visualize profile file: `snakeviz program.prof`

For more details about how to use snakeviz, see [the documentation](https://jiffyclub.github.io/snakeviz/).

#### Memory profiling with memray

If you need to investigate memory use specifically, we use [memray](https://github.com/bloomberg/memray).

> Memray is a memory profiler for Python. It can track memory allocations in Python code, in native extension modules, and in the Python interpreter itself. It can generate several different types of reports to help you analyze the captured memory usage data. While commonly used as a CLI tool, it can also be used as a library to perform more fine-grained profiling tasks.

For more details about how to use memray, see [the documentation](https://bloomberg.github.io/memray/getting_started.html).

## Creating a new release

To create a new release you have to edit the version number in [micro_sam/__version__.py](https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/__version__.py) in a PR. After merging this PR the release will automatically be done by the CI.
