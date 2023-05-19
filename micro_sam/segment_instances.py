import numpy as np
import vigra

from elf.segmentation import embeddings as embed
from segment_anything import SamAutomaticMaskGenerator
from skimage.transform import resize

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

# this is very new, so we don't want to break everything if it's missing
try:
    from elf.segmentation.stitching import stitch_segmentation
except ImportError:
    stitch_segmentation = None


from . import util
from .segment_from_prompts import segment_from_mask


#
# Original SegmentAnything instance segmentation functionality
#


def segment_instances_sam(sam, image, with_background=False, **kwargs):
    segmentor = SamAutomaticMaskGenerator(sam, **kwargs)

    image_ = util._to_image(image)
    masks = segmentor.generate(image_)
    masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    segmentation = np.zeros(image.shape[:2], dtype="uint32")
    for seg_id, mask in enumerate(masks, 1):
        segmentation[mask["segmentation"]] = seg_id

    seg_ids, sizes = np.unique(segmentation, return_counts=True)
    if with_background and 0 not in seg_ids:
        bg_id = seg_ids[np.argmax(seg_ids)]
        segmentation[segmentation == bg_id] = 0
        vigra.analysis.relabelConsecutive(segmentation, out=segmentation)

    return segmentation


#
# Instance segmentation from embeddings
#


def _refine_initial_segmentation(predictor, initial_seg, image_embeddings, i, verbose):
    util.set_precomputed(predictor, image_embeddings, i)

    original_size = image_embeddings["original_size"]
    seg = np.zeros(original_size, dtype="uint32")

    seg_ids = np.unique(initial_seg)
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

    return seg


# This is a first prototype for generating automatic instance segmentations from the image embeddings
# predicted by the segment anything image encoder.

# Main challenge: the larger the image the worse this will get because of the fixed embedding size.
# Ideas:
# - Can we get intermediate, larger embeddings from SAM?
# - Can we run the encoder in a sliding window and somehow stitch the embeddings?
# - Or: run the encoder in a sliding window and stitch the initial segmentation result.
def segment_instances_from_embeddings(
    predictor, image_embeddings, size_threshold=10, i=None,
    offsets=[[-1, 0], [0, -1], [-3, 0], [0, -3]], distance_type="l2", bias=0.0,
    verbose=True, return_initial_seg=False,
):
    util.set_precomputed(predictor, image_embeddings, i)

    embeddings = predictor.get_image_embedding().squeeze().cpu().numpy()
    assert embeddings.shape == (256, 64, 64), f"{embeddings.shape}"
    initial_seg = embed.segment_embeddings_mws(
        embeddings, distance_type=distance_type, offsets=offsets, bias=bias
    ).astype("uint32")
    assert initial_seg.shape == (64, 64), f"{initial_seg.shape}"

    # filter out small objects
    seg_ids, sizes = np.unique(initial_seg, return_counts=True)
    initial_seg[np.isin(initial_seg, seg_ids[sizes < size_threshold])] = 0
    vigra.analysis.relabelConsecutive(initial_seg, out=initial_seg)

    # resize to 256 x 256, which is the mask input expected by SAM
    initial_seg = resize(
        initial_seg, (256, 256), order=0, preserve_range=True, anti_aliasing=False
    ).astype(initial_seg.dtype)
    seg = _refine_initial_segmentation(predictor, initial_seg, image_embeddings, i, verbose)

    if return_initial_seg:
        initial_seg = resize(
            initial_seg, seg.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(seg.dtype)
        return seg, initial_seg
    else:
        return seg


def segment_instances_from_embeddings_with_tiling(
    predictor, image, image_embeddings, tile_shape=(256, 256), tile_overlap=(32, 32),
    size_threshold=10, i=None,
    offsets=[[-1, 0], [0, -1], [-3, 0], [0, -3]], distance_type="l2", bias=0.0,
    verbose=True, return_initial_seg=False,
):
    assert stitch_segmentation is not None

    def segment_tile(input_, block_id=None):

        im = util._to_image(input_)
        print("shape of input:", input_.shape)
        predictor.set_image(im)
        embeddings = predictor.get_image_embedding().squeeze().cpu().numpy()
        assert embeddings.shape == (256, 64, 64), f"{embeddings.shape}"

        seg = embed.segment_embeddings_mws(
            embeddings, distance_type=distance_type, offsets=offsets, bias=bias
        ).astype("uint32")
        assert seg.shape == (64, 64), f"{seg.shape}"

        seg_ids, seg_sizes = np.unique(seg, return_counts=True)
        bg_ids = seg_ids[seg_sizes < size_threshold]
        seg[np.isin(seg, bg_ids)] = 0

        seg = resize(seg, im.shape[:2], order=0, anti_aliasing=False, preserve_range=True).astype(seg.dtype)

        # debugging
        import napari
        from micro_sam.visualization import compute_pca
        embedding_pca = compute_pca(embeddings[None])
        v = napari.view_image(input_)
        v.add_labels(seg)
        v.add_image(embedding_pca, scale=tuple(float(sh) / esh for sh, esh in zip(seg.shape, embedding_pca.shape)))
        napari.run()

        return seg

    initial_seg = stitch_segmentation(image, segment_tile, tile_shape, tile_overlap, shape=image.shape[:2])
    return initial_seg
