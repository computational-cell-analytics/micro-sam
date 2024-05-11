import os
import warnings
from functools import partial

# import urllib
import imageio.v3 as imageio
import numpy as np
import requests
import torch

from hypha_data_store import HyphaDataStore
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

image_url = "https://owncloud.gwdg.de/index.php/s/fSaOJIOYjmFBjPM/download"


def get_sam_model(model_name):
    models = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/staged/1/files/vit_b.pt",
        # TODO
        "vit_b_em_organelles": "",
    }
    model_url = models[model_name]
    checkpoint_path = f"{model_name}.pt"

    if not os.path.exists(checkpoint_path):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = model_name[:5]
    sam = sam_model_registry[model_type]()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sam.load_state_dict(ckpt)
    return sam


def export_onnx_model(
    sam,
    output_path,
    opset: int,
    return_single_mask: bool = True,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
) -> None:

    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules:
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output_path, "wb") as f:
            print(f"Exporting onnx model to {output_path}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


def get_example_image():
    image = imageio.imread(image_url)
    return np.asarray(image)


def _to_image(input_):
    # we require the input to be uint8
    if input_.dtype != np.dtype("uint8"):
        # first normalize the input to [0, 1]
        input_ = input_.astype("float32") - input_.min()
        input_ = input_ / input_.max()
        # then bring to [0, 255] and cast to uint8
        input_ = (input_ * 255).astype("uint8")
    if input_.ndim == 2:
        image = np.concatenate([input_[..., None]] * 3, axis=-1)
    elif input_.ndim == 3 and input_.shape[-1] == 3:
        image = input_
    else:
        raise ValueError(f"Invalid input image of shape {input_.shape}. Expect either 2D grayscale or 3D RGB image.")
    return image


def compute_embeddings(model_name="vit_b"):
    sam = get_sam_model(model_name)
    predictor = SamPredictor(sam)
    image = get_example_image()
    predictor.reset_image()
    predictor.set_image(_to_image(image))
    image_embeddings = predictor.get_image_embedding().cpu().numpy()
    return image_embeddings


async def get_onnx(ds, model_name="vit_b", opset_version=12):
    output_path = f"{model_name}.onnx"
    if not os.path.exists(output_path):
        sam = get_sam_model(model_name)
        export_onnx_model(sam, output_path, opset=opset_version)

    file_id = ds.put("file", f"file://{output_path}", output_path)
    url = ds.get_url(file_id)
    return url


async def start_server():
    from imjoy_rpc.hypha import connect_to_server, login

    server_url = "https://ai.imjoy.io"

    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})

    # Upload to hypha.
    ds = HyphaDataStore()
    await ds.setup(server)

    svc = await server.register_service({
        "name": "Sam Server",
        "id": "bioimageio-colab",
        "config": {
            "visibility": "public"
        },
        "get_onnx": partial(get_onnx, ds=ds),
        "compute_embeddings": compute_embeddings,
        "get_example_image": get_example_image,
        "ping": lambda: "pong"
    })
    sid = svc['id']
    # config_str = f'{{"service_id": "{sid}", "server_url": "{server_url}"}}'
    # encoded_config = urllib.parse.quote(config_str, safe='/', encoding=None, errors=None)
    # annotator_url = 'https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/main/plugins/bioimageio-colab.imjoy.html&config=' + encoded_config
    print(sid)


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    loop.create_task(start_server())

    loop.run_forever()
