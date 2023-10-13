# Using micro_sam on BAND

BAND is a service offered by EMBL Heidelberg that gives access to a virtual desktop for image analysis tasks. It is free to use and `micro_sam` is installed there.
In order to use BAND and start `micro_sam` on it follow these steps:

**Start BAND**
- Go to https://band.embl.de/ and click **Login**. If you have not used BAND before you will need to register for BAND. Currently you can only sign up via a google account.
- Launch a BAND desktop with sufficient resources. It's particularly important to select a GPU. The settings from the image below are a good choice.
- Go to the desktop by clicking **GO TO DESKTOP** in the **Running Desktops** menu. See also the screenshot below.

![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/f965fce2-b924-4fc8-871b-f3201e502138)

**Start micro_sam in BAND**
- Select **Applications->Image Analysis->uSAM** (see screenshot)
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/5daeafb3-119b-4104-8708-aab2960cb21c)
- This will open the micro_sam menu, where you can select the tool you want to use (see screenshot). Note: this may take a few minutes.
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/900ce0b9-4cf8-418c-94f1-e99ac7bc0086)
- For testing if the tool works, it's best to use the **2d annotator** first.
  - You can find an example image to use here: `/scratch/cajal-connectomics/hela-2d-image.png`. Select it via **Select image**. (see screenshot)
![image](https://github.com/computational-cell-analytics/micro-sam/assets/4263537/5fbd1c53-2ba1-47d4-ae50-dfab890ac9d3)
- Then press **2d annotator** and the tool will start.

**Transfering data to BAND**
- TODO
