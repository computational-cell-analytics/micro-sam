# Resource Efficient Finetuning of Segment Anything

All the fullscale experiment in `micro-sam` have been performed on A100s. Can we finetune Segment Anything on limited resources?

## Available Resource Combinations:
- `xps13` (CPU - local)
- `medium` (CPU - SCC)
- `gtx1080`: (GPU - SCC) 8GB
- `rtx5000`: (GPU - SCC) 16GB
- `v100`: (GPU - SCC) 32GB

## Experimental Combinations:
- `vit_t` / `vit_b` (ideally, fewer the parameters, the better for our use-case here)
- number of training images (1 / 2 / 5 / 10)
- number of validation images (fixed to 1)
- number of objects per batch (? - depends on the maximum number which we can fit on the respective resource)

## Inference:

- Using default Segment Anything
- Using `vit_<X>_lm` micro-sam (LM generalist)
- Using finetuned Segment Anything `vit_<X>_covid-if` (training a `covid-if` specialist)
- Using finetuned `vit_<X>_lm` micro-sam (finetuning the LM generalist)

## Training Parameters

Description of parameters which fit the respective resource requirements to run the finetuning experiments

Fixed parameters:
- number of epochs: `100`
- training and validation batch size - `1`
- minimum number of training "samples" for training on the provided images - min. **`50`** (oversample while min. 50 training samples not found) (this is done to avoid the exhaustive time constraints while training with only 1 training sample)
- learning rate: `1e-5`
- optimizer: `Adam`
- lr scheduler: `ReduceLRonPlateau`
- early stopping: `10`
- patch shape: `(512, 512)`
- choice of models: `vit_t` / `vit_b`

### GPU Resources

(32G CPU memory, 8 CPU cores)

1. `gtx1080`:
    - `vit_t`: finetune all layers
        - `n_objects`: 5
    - `vit_b`: freeze `image_encoder`
        - `n_objects`: 10

2. `rtx5000`:
    - `vit_t`: (finetune all layers)
        - `n_objects`: 20
    - `vit_b`: (finetune all layers)
        - `n_objects`: 10

3. `v100`:
    - `vit_t`: (finetune all layers)
        - `n_objects`: 45
    - `vit_b`: (finetune all layers)
        - `n_objects`: 35

### CPU Resources

All jobs are tested on `medium` partition.

1. RAM: 64GB, Cores: 16
    - `vit_b`: finetune all layers
    - `n_objects`: 15 (higher fits, but slows down the training)

2. RAM: 32GB, Cores: 16
    - `vit_b`: finetune all layers
    - `n_objects`: 10 (higher fits, but slows down the training)

3. RAM: 16GB, Cores: 8
    - `vit_t`: finetune all layers
    - `n_objects`: 5

4. RAM: 8GB, Cores: 8
    - `vit_t`: freeze `image_encoder`
    - `n_objects`: 1

5. `XPS13`:
    - TODO

## Scripts:

 TODO: need to explain what are the purpose of the scripts in brief.
