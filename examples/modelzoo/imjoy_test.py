import numpy as np
from imjoy_rpc.hypha import connect_to_server
import time

image = np.random.randint(0, 255, size=(1, 3, 1024, 1024), dtype=np.uint8).astype(
    "float32"
)

# SERVER_URL = 'http://127.0.0.1:9520'  # "https://ai.imjoy.io"
SERVER_URL = "https://hypha.bioimage.io"


async def test_backbone(triton):
    config = await triton.get_config(model_name="micro-sam-vit-b-backbone")
    print(config)

    image = np.random.randint(0, 255, size=(1, 3, 1024, 1024), dtype=np.uint8).astype(
        "float32"
    )

    start_time = time.time()
    result = await triton.execute(
        inputs=[image],
        model_name="micro-sam-vit-b-backbone",
    )
    print("Backbone", result)
    embedding = result['output0__0']
    print("Time taken: ", time.time() - start_time)
    print("Test passed", embedding.shape)


async def run():
    server = await connect_to_server(
        {"name": "test client", "server_url": SERVER_URL, "method_timeout": 100}
    )
    triton = await server.get_service("triton-client")
    await test_backbone(triton)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
