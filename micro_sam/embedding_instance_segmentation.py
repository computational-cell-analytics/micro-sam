import numpy as np

from elf.segmentation import embeddings as embed
from skimage.transform import resize
from tqdm import tqdm

from . import util
from .segment_from_prompts import segment_from_mask


# problem: the larger the image the worse this will get because of the fixed embedding size
# ideas:
# can we get intermediate, larger embeddings from SAM?
# or can we run the encoder in a sliding window and somehow stitch the embeddings?
# (worst case: stitch the initial segmentations)
def automatic_instance_segmentation(
    predictor, image_embeddings, size_threshold=10, i=None,
    offsets=[[-1, 0], [0, -1], [-3, 0], [0, -3]], distance_type="l2", bias=0.0,
    verbose=True, return_initial_seg=False,
):
    util.set_precomputed(predictor, image_embeddings, i)
    embeddings = predictor.get_image_embedding().squeeze().cpu().numpy()
    assert embeddings.shape == (256, 64, 64), f"{embeddings.shape}"
    initial_seg = embed.segment_embeddings_mws(embeddings, distance_type=distance_type, offsets=offsets, bias=bias)
    assert initial_seg.shape == (64, 64), f"{initial_seg.shape}"

    seg_ids, sizes = np.unique(initial_seg, return_counts=True)
    initial_seg[np.isin(initial_seg, seg_ids[sizes < size_threshold])] = 0
    seg_ids = np.unique(initial_seg)

    # resize to 256 x 256, which is the mask input expected by SAM
    initial_seg = resize(
        initial_seg, (256, 256), order=0, preserve_range=True, anti_aliasing=False
    ).astype(initial_seg.dtype)

    original_size = image_embeddings["original_size"]
    seg = np.zeros(original_size, dtype="uint32")

    # TODO be smarter for overlapping masks, (use automatic_mask_generation from SAM as template)
    for seg_id in tqdm(seg_ids[1:], disable=not verbose, desc="Refine masks for automatic instance segmentation"):
        mask = (initial_seg == seg_id)
        assert mask.shape == (256, 256)
        refined = segment_from_mask(predictor,  mask, original_size=original_size).squeeze()
        assert refined.shape == seg.shape
        seg[refined.squeeze()] = seg_id

        # import napari
        # v = napari.Viewer()
        # v.add_image(mask)
        # v.add_labels(refined)
        # napari.run()

    if return_initial_seg:
        initial_seg = resize(
            initial_seg, seg.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(seg.dtype)
        return seg, initial_seg
    else:
        return seg
