import imageio.v3 as imageio
import napari

from micro_sam.instance_segmentation import (
    load_instance_segmentation_with_decoder_from_checkpoint, mask_data_to_segmentation
)
from micro_sam.util import precompute_image_embeddings

checkpoint = "./for_decoder/best.pt"
segmenter = load_instance_segmentation_with_decoder_from_checkpoint(checkpoint, model_type="vit_b")

image_path = "/home/pape/Work/data/incu_cyte/livecell/images/livecell_train_val_images/A172_Phase_A7_1_02d00h00m_1.tif"
image = imageio.imread(image_path)

embedding_path = "./for_decoder/A172_Phase_A7_1_02d00h00m_1.zarr"
image_embeddings = precompute_image_embeddings(
    segmenter._predictor, image, embedding_path,
)
# image_embeddings = None

print("Start segmentation ...")
segmenter.initialize(image, image_embeddings)
masks = segmenter.generate(output_mode="binary_mask")
segmentation = mask_data_to_segmentation(masks, with_background=True)
print("Segmentation done")

v = napari.Viewer()
v.add_image(image)
# v.add_image(segmenter._foreground)
v.add_labels(segmentation)
napari.run()
