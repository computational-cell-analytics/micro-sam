# Installation

There are three ways to install `micro_sam`:
- [From conda](#from-conda) is the recommended way if you want to use all functionality.
- [From source](#from-source) for setting up a development environment to use the latest version and to change and contribute to our software.
- [From installer](#from-installer) to install it without having to use conda (supported platforms: Windows and Linux, supports only CPU). 

You can find more information on the installation and how to troubleshoot it in [the FAQ section](#installation-questions).

We do **not support** installing `micro_sam` with pip.

## From conda

`conda` is a python package manager. If you don't have it installed yet you can follow the instructions [here](https://conda-forge.org/download/) to set it up on your system.
Please make sure that you are using an up-to-date version of conda to install `micro_sam`.
You can also use [mamba](https://mamba.readthedocs.io/en/latest/), which is a drop-in replacement for conda, to install it. In this case, just replace the `conda` command below with `mamba`.

**IMPORTANT**: Do not install `micro_sam` in the base conda environment.

**Installation on Linux and Mac OS:**

`micro_sam` can be installed in an existing environment via:
```bash
conda install -c conda-forge micro_sam
```
or you can create a new environment with it (here called `micro-sam`) via:
```bash
conda create -c conda-forge -n micro-sam micro_sam
```
and then activate it via
```bash
conda activate micro-sam
```

This will also install `pytorch` from the `conda-forge` channel. If you have a recent enough operating system, it will automatically install the best suitable `pytorch` version on your system.
This means it will install the CPU version if you don't have a nVidia GPU, and will install a GPU version if you have.
However, if you have an older operating system, or a CUDA version older than 12, than it may not install the correct version. In this case you will have to specify you're CUDA version, for example for CUDA 11, like this:
```bash
conda install -c conda-forge micro_sam "libtorch=*=cuda11*"
```

**Installation on Windows:**

`pytorch` is currently not available on conda-forge for windows. Thus, you have to install it from the `pytorch` conda channel. In addition, you have to specify two specific dependencies to avoid incompatibilities.
This can be done with the following commands:
```bash
conda install -c pytorch -c conda-forge micro_sam "nifty=1.2.1=*_4" "protobuf<5"
```
to install `micro_sam` in an existing environment and
```bash
conda create -c pytorch -c conda-forge -n micro-sam micro_sam "nifty=1.2.1=*_4" "protobuf<5"
```

## From source

To install `micro_sam` from source, we recommend to first set up an environment with the necessary requirements:
- [environment.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/master/environment.yaml): to set up an environment on Linux or Mac OS.
- [environment_cpu_win.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/master/environment_cpu_win.yaml): to set up an environment on windows with CPU support.
- [environment_gpu_win.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/master/environment_gpu_win.yaml): to set up an environment on windows with GPU support.

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
conda env create -f <ENV_FILE>.yaml
```

4. Activate the environment:

```bash
conda activate sam
```

5. Install `micro_sam`:

```bash
$ pip install -e .
```

## From installer

We also provide installers for Linux and Windows:
- [Linux](https://owncloud.gwdg.de/index.php/s/Fyf57WZuiX1NyXs)
- [Windows](https://owncloud.gwdg.de/index.php/s/ZWrY68hl7xE3kGP)
<!---
- [Mac](https://owncloud.gwdg.de/index.php/s/7YupGgACw9SHy2P)
-->

The installers will not enable you to use a GPU, so if you have one then please consider installing `micro_sam` via [conda](#from-conda) instead. They will also not enable using the python library.

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

<!---
**Mac Installer:**

To use the Mac installer you will need to enable installing unsigned applications. Please follow [the instructions for 'Disabling Gatekeeper for one application only' here](https://disable-gatekeeper.github.io/).

Alternative link on how to disable gatekeeper.
https://www.makeuseof.com/how-to-disable-gatekeeper-mac/

TODO detailed instruction
-->

### Easybuild installation

There is also an easy-build recipe for `micro_sam` under development. You can find more information [here](https://github.com/easybuilders/easybuild-easyconfigs/pull/20636).
