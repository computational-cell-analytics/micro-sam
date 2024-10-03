

def test_example_image():
    from sam_server import get_example_image

    image = get_example_image()
    print(image.shape)


def test_onnx():
    from sam_server import get_sam_model, export_onnx_model

    print("Download!!!")
    sam = get_sam_model("vit_b")
    print("ONNX!!!")
    export_onnx_model(sam, "onnx-test.onnx", opset=12)
    print("Done!!")


def test_embeddings():
    from sam_server import compute_embeddings

    embeds = compute_embeddings()

    print(embeds.shape)


if __name__ == "__main__":
    test_example_image()
    # test_onnx()
    # test_embeddings()
