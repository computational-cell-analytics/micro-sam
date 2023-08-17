import imageio.v3 as imageio
import micro_sam.util as util


# TODO update to hela
def get_image_and_predictor(finetuned_model, with_tiling):
    im = imageio.imread("./data/VID529_F7_1_10d08h00m.tif")

    if finetuned_model:
        checkpoint = "./pdo_vit_h_finetuned.pth"
        embedding_path = "./embeddings/embeddings-pdo-finetuned"
    else:
        checkpoint = None
        embedding_path = "./embeddings/embeddings-pdo-vith"

    if with_tiling:
        tile_shape = (512, 512)
        halo = (128, 128)
        embedding_path += "_tiled.zarr"
    else:
        tile_shape, halo = None, None
        embedding_path += ".zarr"

    predictor = util.get_sam_model(model_type="vit_h", checkpoint_path=checkpoint)
    util.precompute_image_embeddings(predictor, im, embedding_path, tile_shape=tile_shape, halo=halo)

    return im, predictor, embedding_path, tile_shape, halo


def run_annotator():
    from micro_sam.sam_annotator import annotator_2d

    im, predictor, embedding_path, tile_shape, halo = get_image_and_predictor(
        finetuned_model=True, with_tiling=True
    )

    annotator_2d(
        im, embedding_path=embedding_path,
        predictor=predictor, tile_shape=tile_shape, halo=halo,
        precompute_amg_state=True,
    )


if __name__ == "__main__":
    run_annotator()
