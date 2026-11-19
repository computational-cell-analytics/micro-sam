# Installation

There are three ways to install `micro_sam`:
- [From pip](#from-pip) to use all functionality and install via pip.
- [From conda](#from-conda) to use all functionality and install in an encapsulated environment.
- [From source](#from-source) for setting up a development environment to use the latest version and to change and contribute to our software.

You can find more information on the installation and how to troubleshoot it in [the FAQ section](#installation-questions).


## From pip

`pip` is the default python package manager. You can install `micro_sam` via pip with the command:
```bash
pip install micro_sam
```
Note: when using pip you have to manage your python environments yourself. For example, you could create a conda environment and install `micro_sam` via pip in there, or use `venv` for this.
[uv](https://docs.astral.sh/uv/) is a faster alternative to pip that also creates the environment for you:
```bash
uv venv --python 3.12
uv pip install micro_sam
```

Check these points for your system before you install via pip:
- Windows with an NVIDIA GPU: the PyTorch packages on PyPI only support the CPU on Windows. Install PyTorch for your GPU from [pytorch.org](https://pytorch.org/get-started/locally/) first, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`, and then install `micro_sam`.
- Older NVIDIA GPUs (e.g. V100): the default PyTorch packages on PyPI do not support them any more. Install PyTorch from the `cu126` index as described above.
- Mac with an Intel CPU: PyPI has no recent PyTorch packages for these computers. Install `micro_sam` [from conda](#from-conda) instead.

The AnyUp upsampling option of the pixel and object classification tools needs [AnyUp](https://github.com/wimmerth/anyup). It is not available on PyPI or conda-forge, so it has to be installed from GitHub:
```bash
pip install git+https://github.com/wimmerth/anyup
```
To use the SAM1 `vit_t` (MobileSAM) model of the previous version (`micro_sam.v1`), you have to additionally install [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) in the environment where `micro_sam` was installed via pip. It is not available on PyPI, so it has to be installed from GitHub:
```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## From conda

`conda` is a python package manager. We recommend to install it via [Miniforge](https://conda-forge.org/download/), which also provides `mamba`, a faster drop-in replacement for `conda`. You can also look into our [installation FAQs](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#faq) to set it up on your system.
We recommend to use `mamba` from a recent Miniforge (version 25.3.1 or newer) or `micromamba` to install `micro_sam`. `conda` needs a lot more time and memory to install `micro_sam` (see [the installation FAQ](#installation-questions)). If you use `conda`, replace `mamba` by `conda` in the commands below.

**IMPORTANT**: Do not install `micro_sam` in the base conda environment. Create a new environment for it, as described below. Do not use the `defaults` channel of Anaconda or Miniconda, only `conda-forge`.

Create a new environment with `micro_sam` (here called `micro-sam`) with the command for your system:
- Computers without an NVIDIA GPU (including Macs):
```bash
mamba create -n micro-sam -c conda-forge "micro_sam>=2" python=3.12
```
- Linux with an NVIDIA GPU:
```bash
mamba create -n micro-sam -c conda-forge "micro_sam>=2" python=3.12 "pytorch=*=cuda*" "cuda-version=12"
```
- Windows with an NVIDIA GPU:
```bash
mamba create -n micro-sam -c conda-forge "micro_sam>=2" python=3.12 "pytorch=*=cuda*"
```
and then activate it via
```bash
mamba activate micro-sam
```

The NVIDIA GPU commands make sure that a GPU version of PyTorch is installed. Without `"pytorch=*=cuda*"`, conda / mamba can install a PyTorch version that only supports the CPU, even on a computer with a GPU. `"cuda-version=12"` selects a PyTorch version that supports all NVIDIA drivers for CUDA 12 or newer and older GPUs such as the V100.
You can check that `micro_sam` uses the GPU with `micro_sam info`.

`"micro_sam>=2"` makes sure that you get the current version: if the current version cannot be installed together with other packages, e.g. in an existing environment, you get an error instead of an old version.

The AnyUp upsampling option of the pixel and object classification tools needs [AnyUp](https://github.com/wimmerth/anyup). Install it with pip in the conda environment, since it is not available on PyPI or conda-forge:
```bash
pip install git+https://github.com/wimmerth/anyup
```

## From source

To install `micro_sam` from source, we recommend to first set up an environment with the necessary requirements:
- [environment.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/main/environment.yaml): to set up an environment on Windows, Linux or Mac OS.

To create one of these environments and install `micro_sam` into it follow these steps

1. Clone the repository:

```bash
git clone https://github.com/computational-cell-analytics/micro-sam
```

2. Enter it:

```bash
cd micro-sam
```

3. Create the respective environment:

```bash
conda env create -f environment.yaml
```

4. Activate the environment:

```bash
conda activate sam2
```

5. Install `micro_sam`:

```bash
pip install --no-deps -e .
```

The environment already contains all dependencies. `--no-deps` stops pip from installing the PyPI `sam2` on top of conda-forge's `sam-2`.

<!---
I disabled this for now. The installers are very outdated and we shouldn't advertise this feature any more.
## From installer

**Note: The installers are not yet up-to-date with the latest version of micro-sam (v1.4.0 onwards). We recommend installing micro-sam via pip or conda instead.**

We also provide installers for Linux and Windows:
- [Linux](https://owncloud.gwdg.de/index.php/s/Fyf57WZuiX1NyXs)
- [Windows](https://owncloud.gwdg.de/index.php/s/ZWrY68hl7xE3kGP)
- [Mac](https://owncloud.gwdg.de/index.php/s/7YupGgACw9SHy2P)

The installers will not enable you to use a GPU, so if you have one then please consider installing `micro_sam` via [pip](#from-pip) or [conda](#from-conda) instead.

### Linux Installer:

To use the installer:
- Unpack the zip file you have downloaded.
- Make the installer executable: `$ chmod +x micro_sam-1.0.0post0-Linux-x86_64.sh`
- Run the installer: `./micro_sam-1.0.0post0-Linux-x86_64.sh` 
    - You can select where to install `micro_sam` during the installation. By default it will be installed in `$HOME/micro_sam`.
    - The installer will unpack all `micro_sam` files to the installation directory.
- After the installation you can start the annotator with the command `.../micro_sam/bin/napari`.
    - Proceed with the steps described in [Annotation Tools](#annotation-tools)
    - To make it easier to run the annotation tool you can add `.../micro_sam/bin` to your `PATH` or set a softlink to `.../micro_sam/bin/napari`.

### Windows Installer:

- Unpack the zip file you have downloaded.
- Run the installer by double clicking on it.
- Choose installation type: `Just Me(recommended)` or `All Users(requires admin privileges)`.
- Choose installation path. By default it will be installed in `C:\Users\<Username>\micro_sam` for `Just Me` installation or in `C:\ProgramData\micro_sam` for `All Users`.
	- The installer will unpack all micro_sam files to the installation directory.
- After the installation you can start the annotator by double clicking on `.\micro_sam\Scripts\micro_sam.annotator.exe` or  with the command `.\micro_sam\Scripts\napari.exe` from the Command Prompt.
- Proceed with the steps described in [Annotation Tools](#annotation-tools) 

**Mac Installer:**

To use the Mac installer you will need to enable installing unsigned applications. Please follow [the instructions for 'Disabling Gatekeeper for one application only' here](https://disable-gatekeeper.github.io/).

Alternative link on how to disable gatekeeper.
https://www.makeuseof.com/how-to-disable-gatekeeper-mac/

TODO detailed instruction

### Easybuild installation

There is also an easy-build recipe for `micro_sam` under development. You can find more information [here](https://github.com/easybuilders/easybuild-easyconfigs/pull/20636).
-->
