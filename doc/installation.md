# Installation

There are three ways to install `micro_sam`:
- [From mamba](#from-mamba) is the recommended way if you want to use all functionality.
- [From source](#from-source) for setting up a development environment to use the latest version and to change and contribute to our software.
- [From installer](#from-installer) to install it without having to use mamba (supported platforms: Windows and Linux, supports only CPU). 

You can find more information on the installation and how to troubleshoot it in [the FAQ section](#installation-questions).

We do *not* recommend installing `micro-sam` with pip.

## From mamba

[mamba](https://mamba.readthedocs.io/en/latest/) is a drop-in replacement for conda, but much faster.
The steps below may also work with `conda`, but we recommend using `mamba`, especially if the installation does not work with `conda`.
You can follow the instructions [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install `mamba`.

**IMPORTANT**: Make sure to avoid installing anything in the base environment.

`micro_sam` can be installed in an existing environment via:
```bash
$ mamba install -c conda-forge micro_sam
```
or you can create a new environment (here called `micro-sam`) via:
```bash
$ mamba create -c conda-forge -n micro-sam micro_sam
```
if you want to use the GPU you need to install PyTorch from the `pytorch` channel instead of `conda-forge`. For example:
```bash
$ mamba create -c pytorch -c nvidia -c conda-forge -n micro-sam micro_sam pytorch pytorch-cuda=12.1
```
You may need to change this command to install the correct CUDA version for your system, see [https://pytorch.org/](https://pytorch.org/) for details.


## From source

To install `micro_sam` from source, we recommend to first set up an environment with the necessary requirements:
- [environment_gpu.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/master/environment_gpu.yaml): sets up an environment with GPU support.
- [environment_cpu.yaml](https://github.com/computational-cell-analytics/micro-sam/blob/master/environment_cpu.yaml): sets up an environment with CPU support.

To create one of these environments and install `micro_sam` into it follow these steps

1. Clone the repository:

```bash
$ git clone https://github.com/computational-cell-analytics/micro-sam
```

2. Enter it:

```bash
$ cd micro-sam
```

3. Create the GPU or CPU environment:

```bash
$ mamba env create -f <ENV_FILE>.yaml
```

4. Activate the environment:

```bash
$ mamba activate sam
```

5. Install `micro_sam`:

```bash
$ pip install -e .
```

## From installer

We also provide installers for Linux and Windows:
- [Linux](https://owncloud.gwdg.de/index.php/s/nrNBuHr9ncJqid6)
- [Windows](https://owncloud.gwdg.de/index.php/s/kZmpAIBDmUSu4e9)
<!---
- [Mac](https://owncloud.gwdg.de/index.php/s/7YupGgACw9SHy2P)
-->

The installers will not enable you to use a GPU, so if you have one then please consider installing `micro_sam` via [mamba](#from-mamba) instead. They will also not enable using the python library.

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
