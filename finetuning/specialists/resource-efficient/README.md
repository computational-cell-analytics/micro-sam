# Resource Efficient Finetuning of Segment Anything

All the fullscale experiment in `micro-sam` have been performed on A100s.

Question: Can we finetune Segment Anything on limited resources?

TLDR: Finetuning ViT Base (`vit_b`) is the best bet on most workstation / cluster-level GPUs. Reduce the number of objects per batch to fit to your desired resource. Feel free to read ahead if you want more specifics on this, or let us know for further discussion (see our [documentation](https://computational-cell-analytics.github.io/micro-sam/) for more details on this)

## Available Resource Combinations:
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

(32GB CPU memory, 8 CPU cores)

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

## Scripts:

- `check_training_times.py`: The scripts to check the time taken to achieve the best model. The reported times are menioned in [results](#results) below.
- `covid_if_finetuning.py`: The finetuning scripts for segmenting cells in immunofluorescence data.
- `plot_experiments.py`: The scripts for plotting the quantitative results for the resource-efficient finetuning experiments.
- `run_evaluations.py`: The scripts to run quantitative evaluation for different resource efficient finetuned SAM models.
- `run_resource_efficient_finetuning,py`: Convenience scripts for submitting batch jobs via slurm to HLRN for finetuning SAM on Covid IF.



 ## Results:

| Resource | Finetuned Model        | Number of Images | Best Epoch | Train Time *(in s)* |
|----------|------------------------|------------------|------------|------------|
| v100     | vit_b (freeze None)    | 1                | 9          | 752.39     |
| v100     | vit_b (freeze None)    | 2                | 26         | 2051.77    |
| v100     | vit_b (freeze None)    | 5                | 21         | 1653.99    |
| v100     | vit_b (freeze None)    | 10               | 39         | 2998.08    |
| v100     | vit_b_lm (freeze None) | 1                | 24         | 1874.83    |
| v100     | vit_b_lm (freeze None) | 2                | 42         | 3205.59    |
| v100     | vit_b_lm (freeze None) | 5                | 42         | 3196.15    |
| v100     | vit_b_lm (freeze None) | 10               | 34         | 2612.99    |
| rtx5000  | vit_b (freeze None)    | 1                | 17         | 1192.79    |
| rtx5000  | vit_b (freeze None)    | 2                | 10         | 725.15     |
| rtx5000  | vit_b (freeze None)    | 5                | 56         | 3759.01    |
| rtx5000  | vit_b (freeze None)    | 10               | 36         | 2427.17    |
| rtx5000  | vit_b_lm (freeze None) | 1                | 31         | 2089.22    |
| rtx5000  | vit_b_lm (freeze None) | 2                | 24         | 1622.69    |
| rtx5000  | vit_b_lm (freeze None) | 5                | 53         | 3477.83    |
| rtx5000  | vit_b_lm (freeze None) | 10               | 28         | 1869.33    |
| gtx1080  | vit_b (freeze image_encoder)    | 1                | 22         | 2629.69    |
| gtx1080  | vit_b (freeze image_encoder)    | 2                | 22         | 2664.08    |
| gtx1080  | vit_b (freeze image_encoder)    | 5                | 12         | 1523.38    |
| gtx1080  | vit_b (freeze image_encoder)    | 10               | 49         | 5858.78    |
| gtx1080  | vit_b_lm (freeze image_encoder) | 1                | 18         | 2186.33    |
| gtx1080  | vit_b_lm (freeze image_encoder) | 2                | 13         | 1608.46    |
| gtx1080  | vit_b_lm (freeze image_encoder) | 5                | 23         | 2762.22    |
| gtx1080  | vit_b_lm (freeze image_encoder) | 10               | 22         | 2617.61    |
| cpu32g  | vit_b (freeze None)    | 1                 | 5          | 6302.03    |
| cpu32g  | vit_b (freeze None)    | 2                 | 27         | 29153.65   |
| cpu32g  | vit_b (freeze None)    | 5                 | 46         | 53502.85   |
| cpu32g  | vit_b (freeze None)    | 10                | 25         | 20885.33   |
| cpu32g  | vit_b_lm (freeze None) | 1                 | 27         | 21711.23   |
| cpu32g  | vit_b_lm (freeze None) | 2                 | 35         | 34443.09   |
| cpu32g  | vit_b_lm (freeze None) | 5                 | 25         | 32750.22   |
| cpu32g  | vit_b_lm (freeze None) | 10                | 22         | 19229.84   |
| cpu64g  | vit_b (freeze None)    | 1                 | 12         | 11439.01   |
| cpu64g  | vit_b (freeze None)    | 2                 | 23         | 26225.69   |
| cpu64g  | vit_b (freeze None)    | 5                 | 21         | 18675.01   |
| cpu64g  | vit_b (freeze None)    | 10                | 43         | 50894.71   |
| cpu64g  | vit_b_lm (freeze None) | 1                 | 25         | 23291.25   |
| cpu64g  | vit_b_lm (freeze None) | 2                 | 41         | 40262.73   |
| cpu64g  | vit_b_lm (freeze None) | 5                 | 33         | 33137.21   |
| cpu64g  | vit_b_lm (freeze None) | 10                | 37         | 47490.61   |
