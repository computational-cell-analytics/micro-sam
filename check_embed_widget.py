import napari
from skimage.data import astronaut
from micro_sam.sam_annotator._state import SamState

x = astronaut()

v = napari.Viewer()
v.add_image(x)


@v.bind_key("p")
def print_embed(v):
    state = SamState()
    print("Image Embeddings are None:", state.image_embeddings is None)


napari.run()
