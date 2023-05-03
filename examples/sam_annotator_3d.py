import z5py
from micro_sam.sam_annotator import annotator_3d


def _load_raw(input_path, halo=[40, 384, 384], center=None):
    with z5py.File(input_path, "r") as f:
        ds = f["raw"]
        shape = ds.shape
        center = [sh // 2 for sh in shape] if center is None else center
        bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
        raw = ds[bb]
    print("Loaded raw from bb:", bb)
    return raw


def main():
    halo = [40, 384, 384]
    input_path = "/home/pape/Work/data/moebius/mito/4007.n5"
    raw = _load_raw(input_path, halo)
    embedding_path = "./embeddings/embeddings-carving-mito.zarr"
    annotator_3d(raw, embedding_path, show_embeddings=False)


if __name__ == "__main__":
    main()
