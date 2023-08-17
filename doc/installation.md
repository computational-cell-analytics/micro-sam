# Installation

`micro_sam` requires the following dependencies:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [SegmentAnything](https://github.com/facebookresearch/segment-anything#installation)
- [elf](https://github.com/constantinpape/elf)
- [napari](https://napari.org/stable/) (for the interactive annotation tools)
- [torch_em](https://github.com/constantinpape/torch-em) (for the training functionality)

## From conda

`micro_sam` is available as a conda package and can be installed via
```
$ conda install -c conda-forge micro_sam
```

This command will only install the `micro_sam` library without the required dependencies for the annotation functionality and training.
To use the annotation functionality you also need to install `napari`:
```
$ conda install -c conda-forge napari pyqt
```
And to use the training functionality `torch_em`:
```
$ conda install -c conda-forge torch_em
```

In case the installation via conda takes too long consider using [mamba](https://mamba.readthedocs.io/en/latest/).
Once you have it installed you can simply replace the `conda` commands with `mamba`.

## From installer

We also provide installers for Linux, Mac and Windows:
- [Linux](https://owncloud.gwdg.de/index.php/s/HRp948SDkaWzCuV)
- [Mac](https://owncloud.gwdg.de/index.php/s/HpGzlXrgJ8VDgnI)
- [Windows](https://owncloud.gwdg.de/index.php/s/BVipOmDPR2TXmxk)

Note that these installers are stil experimental and not yet fully tested.
If you encounter problems with them then please consider installing `micro_sam` via [conda](#from-conda) instead.

**Linux Installer:**

To use the installer:
- Unpack the you have downloaded.
- Make the installer executable: `$ chmod +x micro_sam-0.2.0post1-Linux-x86_64.sh`
- Run the installer: `$./micro_sam-0.2.0post1-Linux-x86_64.sh$` 
    - You can select where to install `micro_sam` during the installation. By default it will be installed in `$HOME/micro_sam`.
    - The installer will unpack all `micro_sam` files to the installation directory.
- After the installation you can start the annotator with the command `.../micro_sam/bin/micro_sam.annotator`.
    - To make it easier to run the annotation tool you can add `.../micro_sam/bin` to your `PATH` or set a softlink to `.../micro_sam/bin/micro_sam.annotator`.

**Mac Installer:**

To use the Mac installer you will need to enable installing unsigned applications. Please follow [the instructions for 'Disabling Gatekeeper for one application only' here](https://disable-gatekeeper.github.io/).
<!---
Alternative link on how to disable gatekeeper.
https://www.makeuseof.com/how-to-disable-gatekeeper-mac/
-->

TODO detailed instruction

**Windows Installer:**

TODO

## From source

To install `micro_sam` from source, we recommend to first set up a conda environment with the necessary requirements:
- [environment_gpu.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/master/environment_gpu.yaml): sets up an environment with GPU support.
- [environment_cpu.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/master/environment_cpu.yaml): sets up an environment with CPU support.

To create one of these environments and install `micro_sam` into it follow these steps

1. Clone the repository:
```
$ git clone https://github.com/computational-cell-analytics/micro-sam
```
2. Enter it:
```
$ cd micro_sam
```
3. Create the GPU or CPU environment:

```
$ conda env create -f <ENV_FILE>.yaml
```
4. Activate the environment:
```
$ conda activate sam
```
5. Install `micro_sam`:
```
$ pip install -e .
```

**Troubleshooting:**

- On some systems `conda` is extremely slow and cannot resolve the environment in the step `conda env create ...`. You can use `mamba` instead, which is a faster re-implementation of `conda`. It can resolve the environment in less than a minute on any system we tried. Check out [this link](https://mamba.readthedocs.io/en/latest/installation.html) for how to install `mamba`. Once you have installed it, run `mamba env create -f <ENV_FILE>.yaml` to create the env.
- Installation on MAC with a M1 or M2 processor:
    - The pytorch installation from `environment_cpu.yaml` does not work with a MAC that has an M1 or M2 processor. Instead you need to:
        - Create a new environment: `mamba create -c conda-forge python pip -n sam`
        - Activate it va `mamba activate sam`
        - Follow the instructions for how to install pytorch for MAC via conda from [pytorch.org](https://pytorch.org/).
        - Install additional dependencies: `mamba install -c conda-forge napari python-elf tqdm`
        - Install SegmentAnything: `pip install git+https://github.com/facebookresearch/segment-anything.git`
        - Install `micro_sam` by running `pip install -e .` in this folder.
    - **Note:** we have seen many issues with the pytorch installation on MAC. If a wrong pytorch version is installed for you (which will cause pytorch errors once you run the application) please try again with a clean `mambaforge` installation. Please install the `OS X, arm64` version from [here](https://github.com/conda-forge/miniforge#mambaforge).
    - Some MACs require a specific installation order of packages. If the steps layed out above don't work for you please check out the procedure described [in this github issue](https://github.com/computational-cell-analytics/micro-sam/issues/77).
