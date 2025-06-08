# Using `micro_sam` on BAND

BAND is a service offered by "The German Network for Bioinformatics Infrastructure" (de.NBI) that gives access to a virtual desktop for image analysis tasks. It is free to use and `micro_sam` is installed there.
In order to use BAND and start `micro_sam` on it follow these steps:

## Start BAND
- Go to https://bandv1.denbi.uni-tuebingen.de/ and click **Login**. If you have not used BAND before you will need to register for BAND. Currently you can only sign up via a Google account. NOTE: It takes a couple of seconds for the "Launch Desktops" window to appear.
- Launch a BAND desktop with sufficient resources. It's particularly important to select a GPU. The settings from the image below are a good choice.
- Go to the desktop by clicking **GO TO DESKTOP** in the **Running Desktops** menu. See also the screenshot below.

![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/f965fce2-b924-4fc8-871b-f3201e502138)
<!-- TODO: Change the screenshot here to match the latest UI -->

## Start `micro_sam` in BAND
- Select **Applications -> Image Analysis -> microSAM** (see screenshot)
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/5daeafb3-119b-4104-8708-aab2960cb21c)
<!-- TODO: Change the screenshot here to match the latest UI -->

- This will open the napari GUI, where you can select the images and annotator tools you want to use (see screenshot). NOTE: this may take a few minutes.
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/900ce0b9-4cf8-418c-94f1-e99ac7bc0086)
<!-- TODO: Change the screenshot here to match the latest UI -->

- For testing if the tool works, it's best to use the **Annotator 2d** first.
  - You can find an example image to use by selection `File` -> `Open Sample` -> `Segment Anything for Microscopy` -> `HeLa 2d example data` (see screenshot)
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/5fbd1c53-2ba1-47d4-ae50-dfab890ac9d3)

<!-- TODO: Change the screenshot here to match the latest UI -->
- Then select `Plugins` -> `Segment Anything for Microscopy` -> `Annotator 2d`, and the tool will start.

## Transfering data to BAND

To copy data to and from BAND you can use any cloud storage, e.g. ownCloud, dropbox or google drive. For this, it's important to note that copy and paste, which you may need for accessing links on BAND, works a bit different in BAND:
- To copy text into BAND you first need to copy it on your computer (e.g. via selecting it + `Ctrl + C`).
- Then go to the browser window with BAND and press `Ctrl + Shift + Alt`. This will open a side window where you can paste your text via `Ctrl + V`.
- Then select the text in this window and copy it via `Ctrl + C`.
- Now you can close the side window via `Ctrl + Shift + Alt` and paste the text in band via `Ctrl + V`

The video below shows how to copy over a link from owncloud and then download the data on BAND using copy and paste:

https://github.com/computational-cell-analytics/micro-sam/assets/4263537/825bf86e-017e-41fc-9e42-995d21203287
