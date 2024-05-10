# Using `micro_sam` on BAND

BAND is a service offered by EMBL Heidelberg that gives access to a virtual desktop for image analysis tasks. It is free to use and `micro_sam` is installed there.
In order to use BAND and start `micro_sam` on it follow these steps:

## Start BAND
- Go to https://band.embl.de/ and click **Login**. If you have not used BAND before you will need to register for BAND. Currently you can only sign up via a Google account.
- Launch a BAND desktop with sufficient resources. It's particularly important to select a GPU. The settings from the image below are a good choice.
- Go to the desktop by clicking **GO TO DESKTOP** in the **Running Desktops** menu. See also the screenshot below.

![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/f965fce2-b924-4fc8-871b-f3201e502138)

## Start `micro_sam` in BAND
- Select **Applications -> Image Analysis -> uSAM** (see screenshot)
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/5daeafb3-119b-4104-8708-aab2960cb21c)
- This will open the micro_sam menu, where you can select the tool you want to use (see screenshot). Note: this may take a few minutes.
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/900ce0b9-4cf8-418c-94f1-e99ac7bc0086)
- For testing if the tool works, it's best to use the **2d annotator** first.
  - You can find an example image to use here: `/scratch/cajal-connectomics/hela-2d-image.png`. Select it via **Select image**. (see screenshot)
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/5fbd1c53-2ba1-47d4-ae50-dfab890ac9d3)
- Then press **2d annotator** and the tool will start.

## Transfering data to BAND

To copy data to and from BAND you can use any cloud storage, e.g. ownCloud, dropbox or google drive. For this, it's important to note that copy and paste, which you may need for accessing links on BAND, works a bit different in BAND:
- To copy text into BAND you first need to copy it on your computer (e.g. via selecting it + `Ctrl + C`).
- Then go to the browser window with BAND and press `Ctrl + Shift + Alt`. This will open a side window where you can paste your text via `Ctrl + V`.
- Then select the text in this window and copy it via `Ctrl + C`.
- Now you can close the side window via `Ctrl + Shift + Alt` and paste the text in band via `Ctrl + V`

The video below shows how to copy over a link from owncloud and then download the data on BAND using copy and paste:

https://github.com/computational-cell-analytics/micro-sam/assets/4263537/825bf86e-017e-41fc-9e42-995d21203287
