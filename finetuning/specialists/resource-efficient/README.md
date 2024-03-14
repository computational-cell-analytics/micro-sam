# Resource Efficient Finetuning

a. Resource combinations:
    - `xps13` (CPU compute) (local)
    - `medium` (CPU compute partition) (SCC)
    - `gtx1080`: 8GB (SCC)
    - `rtx5000`: 16GB (SCC)
    - `v100`: 32GB (SCC)

b. Experiment combinations:
    i. `vit_t` / `vit_b` / both
    ii. number of training images (1/2/5/10)
    iii. number of validation images (set to 1)
    iv. number of objects per batch (X to fit all)
    v. number of epochs for different number of training samples
        - idea: let's keep the number of epochs per combination training to stay the same (for consistency in early stopping, lr scheduler)

c. **Inference**:
    i. vanilla SAM
    ii. `vit_<X>_lm` micro-sam (using finetuned LM generalist)
    iii. `vit_<X>_covid-if` micro-sam (training a specialist)
    iv. finetuning `vit_<X>_lm` microsam (finetuning the LM generalist)

## Combinations

Description of parameters which fit the resource requirements to run the finetuning experiments

Fixed parameters:
- number of epochs: 100
- train and val batch size - 1
- minimum number of training "samples" for training on the provided images - min. **50** (oversample while we don't find min. 50 training samples)
- learning rate: 1e-5
- optimizer: Adam
- lr scheduler: ReduceLRonPlateau
- early stopping: 10
- patch shape: (512, 512)
- choice of models: vit_t / vit_b

### GPU Resources

(32G cpu memory, 8 cpu cores)

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

All jobs are tested on `medium` partition

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

TODO: check on XPS13
