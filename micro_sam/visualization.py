import numpy as np
from elf.segmentation.embeddings import embedding_pca


#
# PCA visualization for the image embeddings
#

def compute_pca(embeddings):
    """Compute the pca projection of the embeddings to visualize them as RGB image.
    """
    if embeddings.ndim == 4:
        pca = embedding_pca(embeddings.squeeze()).transpose((1, 2, 0))
    elif embeddings.ndim == 5:
        pca = []
        for embed in embeddings:
            vis = embedding_pca(embed.squeeze()).transpose((1, 2, 0))
            pca.append(vis)
        pca = np.stack(pca)
    else:
        raise ValueError(f"Expect input of ndim 4 or 5, got {embeddings.ndim}")
    return pca


def project_embeddings_for_visualization(embeddings, shape):
    assert embeddings.ndim == len(shape) + 2, f"{embeddings.shape}, {shape}"

    def get_crop(embedding_vis, shape):
        if shape[0] == shape[1]:  # square image, we don't need to do anything
            crop = np.s_[:, :]
        else:  # non-square image, we need to crop away the padding in the shorter axis
            if shape[0] > shape[1]:
                aspect_ratio = float(shape[1] / shape[0])
                crop = np.s_[:, :int(aspect_ratio * embedding_vis.shape[0])]
            else:
                aspect_ratio = float(shape[0] / shape[1])
                crop = np.s_[:int(aspect_ratio * embedding_vis.shape[1]), :]
        return crop

    embedding_vis = compute_pca(embeddings)
    if len(shape) == 2:
        crop, scale = get_crop(embedding_vis, shape)
        embedding_vis = embedding_vis[crop]
    elif len(shape) == 3:
        crop = get_crop(embedding_vis[0], shape[1:])
        crop = (slice(None),) + crop
        embedding_vis = embedding_vis[crop]
    else:
        raise ValueError(f"Expect 2d or 3d data, got {len(shape)}")
    scale = tuple(float(sh) / vsh for sh, vsh in zip(shape, embedding_vis.shape))
    return embedding_vis, scale
