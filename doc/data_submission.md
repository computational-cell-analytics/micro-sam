# Community Data Submissions

We are looking to further improve the `micro_sam` models by training on more diverse microscopy data.
For this, we want to collect data where the models don't work well yet, and need your help!

If you are using `micro_sam` for a task where the current models don't do a good job, but you have annotated data and successfully fine-tuned a model, then you can submit this data to us, so that we can use it to train our next version of improved microscopy models.
To do this, please either create an [issue on github](https://github.com/computational-cell-analytics/micro-sam/issues) or a post on [image.sc](https://forum.image.sc/) and:
- Use a title "Data submission for micro_sam: ..." ("..." should be a title for your data, e.g. "cells in brightfield microscopy")
    - On [image.sc](https://forum.image.sc/) use the tag `micro-sam`.
- Briefly describe your data and add an image that shows the microscopy data and the segmentation masks you have.
- Make sure to describe:
    - The imaging modality and the structure(s) that you have segmented.
    - The `micro_sam` model you have used for finetuning and segmenting the data.
        - You can also submit data that was not segmented with `micro_sam`, as long as you have sufficient annotations we are happy to include it!
    - How many images and annotations you have / can submit and how you have created the annotations.
        - You should submit at least 5 images / 100 annotated objects to have a meaningful impact. If you are unsure if you have enough data please go ahead and create the issue / post and we can discuss the details.
    - Which data-format your images and annotations are stored in. We recommend using either `tif` images or `ome.zarr` files.
- Please indicate that you are willing to share the data for training purpose (see also next paragraph).

Once you have created the post / issue, we will check if your data is suitable for submission or discuss with you how it could be extended to be suitable. Then:
- We will share an agreement for data sharing. You can find **a draft** [here](https://docs.google.com/document/d/1X3VOf1qtJ5WtwDGcpGYZ-kfr3E2paIEquyuCtJnF_I0/edit?usp=sharing).
- You will be able to choose how you want to submit / publish your data.
    - Share it under a CC0 license. In this case, we will use the data for re-training and also make it publicly available as soon as the next model versions become available.
    - Share it for training with the option to publish it later. For example, if your data is unpublished and you want to only published once the respective publication is available. In this case, we will use the data for re-training, but not make it freely available yet. We will check with you peridiodically to see if your data can now be published.
    - Share it for training only. In this case, we will re-train the model on it, but not make it publicly available.
- We encourage you to choose the first option (making the data available under CC0).
- We will then send you a link to upload your data, after you have agreed to these terms.
