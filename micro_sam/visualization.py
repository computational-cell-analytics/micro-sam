import numpy as np
from elf.segmentation.embeddings import embedding_pca


def compute_pca(embeddings):
    """Compute the pca projection of the embeddings to visualize them as RGB image.
    """
    if embeddings.ndim == 3:
        pca = embedding_pca(embeddings.squeeze()).transpose((1, 2, 0))
    elif embeddings.ndim == 4:
        pca = []
        for embed in embeddings:
            vis = embedding_pca(embed.squeeze()).transpose((1, 2, 0))
            pca.append(vis)
        pca = np.stack(pca)
    else:
        raise ValueError(f"Expect input of ndim 3 or 4, gout {embeddings.ndim}")
    return pca
