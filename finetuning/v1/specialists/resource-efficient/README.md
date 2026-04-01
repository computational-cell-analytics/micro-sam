# Resource Efficient Finetuning of Segment Anything

All the fullscale experiment in `micro-sam` have been performed on A100s.

Question: Can we finetune Segment Anything on limited resources?

TLDR: Finetuning ViT Base (`vit_b`) is the best bet on most workstation / cluster-level GPUs. Reduce the number of objects per batch to fit to your desired resource. Feel free to read ahead if you want more specifics on this, or let us know for further discussion (see our [documentation](https://computational-cell-analytics.github.io/micro-sam/) for more details on this)

## Available Resource Combinations:
- `medium` (CPU - SCC)
- `GTX1080`: (GPU - SCC) 8GB
- `RTX5000`: (GPU - SCC) 16GB
- `V100`: (GPU - SCC) 32GB

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
- optimizer: `AdamW`
- lr scheduler: `ReduceLRonPlateau`
- early stopping: `10`
- patch shape: `(512, 512)`
- choice of models: `vit_t` / `vit_b` / `vit_t_lm` / `vit_b_lm`

### GPU Resources

(32GB CPU memory, 8 CPU cores)

1. `GTX1080`:
    - `vit_t`: finetune all layers
        - `n_objects`: 5
    - `vit_b`: freeze `image_encoder`
        - `n_objects`: 10

2. `RTX5000`:
    - `vit_t`: (finetune all layers)
        - `n_objects`: 20
    - `vit_b`: (finetune all layers)
        - `n_objects`: 10

3. `V100`:
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



<!-- NOTE: The commented results below are from "v2" version of `micro-sam`.-->
 <!-- ## Results:

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
| cpu64g  | vit_b_lm (freeze None) | 10                | 37         | 47490.61   | -->


 ## Results:

| Resource | Finetuned Model                   | Number of Images | Best Epoch | Train Time |
|----------|-----------------------------------|------------------|------------|------------|
| V100     | vit_b (Full Finetuning)           | 1                | 3          | 0:05:07    |
| V100     | vit_b (Full Finetuning)           | 2                | 10         | 0:14:01    |
| V100     | vit_b (Full Finetuning)           | 5                | 10         | 0:14:09    |
| V100     | vit_b (Full Finetuning)           | 10               | 20         | 0:26:24    |
| V100     | vit_b (LoRA)                      | 1                | 32         | 0:39:32    |
| V100     | vit_b (LoRA)                      | 2                | 58         | 1:10:25    |
| V100     | vit_b (LoRA)                      | 5                | 13         | 0:16:40    |
| V100     | vit_b (LoRA)                      | 10               | 42         | 0:51:10    |
| V100     | vit_b_lm (Full Finetuning)        | 1                | 1          | 0:02:33    |
| V100     | vit_b_lm (Full Finetuning)        | 2                | 4          | 0:06:19    |
| V100     | vit_b_lm (Full Finetuning)        | 5                | 12         | 0:16:14    |
| V100     | vit_b_lm (Full Finetuning)        | 10               | 2          | 0:03:48    |
| V100     | vit_b_lm (LoRA)                   | 1                | 8          | 0:10:45    |
| V100     | vit_b_lm (LoRA)                   | 2                | 23         | 0:28:33    |
| V100     | vit_b_lm (LoRA)                   | 5                | 22         | 0:27:23    |
| V100     | vit_b_lm (LoRA)                   | 10               | 5          | 0:07:11    |
| RTX5000  | vit_b (Full Finetuning)           | 1                | 13         | 0:15:09    |
| RTX5000  | vit_b (Full Finetuning)           | 2                | 13         | 0:15:00    |
| RTX5000  | vit_b (Full Finetuning)           | 5                | 20         | 0:22:29    |
| RTX5000  | vit_b (Full Finetuning)           | 10               | 43         | 0:46:55    |
| RTX5000  | vit_b (LoRA)                      | 1                | 46         | 0:48:30    |
| RTX5000  | vit_b (LoRA)                      | 2                | 23         | 0:24:53    |
| RTX5000  | vit_b (LoRA)                      | 5                | 39         | 0:41:14    |
| RTX5000  | vit_b (LoRA)                      | 10               | 16         | 0:17:37    |
| RTX5000  | vit_b_lm (Full Finetuning)        | 1                | 4          | 0:05:26    |
| RTX5000  | vit_b_lm (Full Finetuning)        | 2                | 4          | 0:05:25    |
| RTX5000  | vit_b_lm (Full Finetuning)        | 5                | 3          | 0:04:21    |
| RTX5000  | vit_b_lm (Full Finetuning)        | 10               | 3          | 0:04:22    |
| RTX5000  | vit_b_lm (LoRA)                   | 1                | 15         | 0:16:37    |
| RTX5000  | vit_b_lm (LoRA)                   | 2                | 26         | 0:28:03    |
| RTX5000  | vit_b_lm (LoRA)                   | 5                | 22         | 0:23:54    |
| RTX5000  | vit_b_lm (LoRA)                   | 10               | 32         | 0:34:04    |
| GTX1080  | vit_b (Freeze `image_encoder`)    | 1                | 6          | 0:13:39    |
| GTX1080  | vit_b (Freeze `image_encoder`)    | 2                | 3          | 0:07:55    |
| GTX1080  | vit_b (Freeze `image_encoder`)    | 5                | 26         | 0:51:34    |
| GTX1080  | vit_b (Freeze `image_encoder`)    | 10               | 40         | 1:18:05    |
| GTX1080  | vit_b_lm (Freeze `image_encoder`) | 1                | 10         | 0:21:30    |
| GTX1080  | vit_b_lm (Freeze `image_encoder`) | 2                | 2          | 0:06:15    |
| GTX1080  | vit_b_lm (Freeze `image_encoder`) | 5                | 7          | 0:15:05    |
| GTX1080  | vit_b_lm (Freeze `image_encoder`) | 10               | 13         | 0:15:05    |
| CPU (32G)  | vit_b (Full Finetuning)         | 1                | 15         | 3:48:52    |
| CPU (32G)  | vit_b (Full Finetuning)         | 2                | 18         | 4:36:06    |
| CPU (32G)  | vit_b (Full Finetuning)         | 5                | 30         | 7:47:20    |
| CPU (32G)  | vit_b (Full Finetuning)         | 10               | 24         | 5:41:31    |
| CPU (32G)  | vit_b (LoRA)                    | 1                | 26         | 5:21:23    |
| CPU (32G)  | vit_b (LoRA)                    | 2                | 12         | 2:53:41    |
| CPU (32G)  | vit_b (LoRA)                    | 5                | 50         | 11:03:15   |
| CPU (32G)  | vit_b (LoRA)                    | 10               | 13         | 2:57:08    |
| CPU (32G)  | vit_b_lm (Full Finetuning)      | 1                | 3          | 0:55:36    |
| CPU (32G)  | vit_b_lm (Full Finetuning)      | 2                | 24         | 5:43:28    |
| CPU (32G)  | vit_b_lm (Full Finetuning)      | 5                | 1          | 0:16:03    |
| CPU (32G)  | vit_b_lm (Full Finetuning)      | 10               | 6          | 2:01:30    |
| CPU (32G)  | vit_b_lm (LoRA)                 | 1                | 15         | 3:25:33    |
| CPU (32G)  | vit_b_lm (LoRA)                 | 2                | 9          | 2:58:05    |
| CPU (32G)  | vit_b_lm (LoRA)                 | 5                | 14         | 3:31:14    |
| CPU (32G)  | vit_b_lm (LoRA)                 | 10               | 7          | 1:58:57    |
| CPU (64G)  | vit_b (Full Finetuning)        | 1                 | 6          | 3:20:00    |
| CPU (64G)  | vit_b (Full Finetuning)        | 2                 | 15         | 4:23:10    |
| CPU (64G)  | vit_b (Full Finetuning)        | 5                 | 16         | 4:05:15    |
| CPU (64G)  | vit_b (Full Finetuning)        | 10                | 15         | 3:51:02    |
| CPU (64G)  | vit_b (LoRA)                   | 1                 | 27         | 6:20:52    |
| CPU (64G)  | vit_b (LoRA)                   | 2                 | 46         | 19:51:34   |
| CPU (64G)  | vit_b (LoRA)                   | 5                 | 29         | 8:01:34    |
| CPU (64G)  | vit_b (LoRA)                   | 10                | 19         | 5:20:02    |
| CPU (64G)  | vit_b_lm (Full Finetuning)     | 1                 | 3          | 1:44:35    |
| CPU (64G)  | vit_b_lm (Full Finetuning)     | 2                 | 10         | 2:57:22    |
| CPU (64G)  | vit_b_lm (Full Finetuning)     | 5                 | 8          | 2:31:04    |
| CPU (64G)  | vit_b_lm (Full Finetuning)     | 10                | 5          | 1:28:26    |
| CPU (64G)  | vit_b_lm (LoRA)                | 1                 | 16         | 4:39:26    |
| CPU (64G)  | vit_b_lm (LoRA)                | 2                 | 1          | 0:19:46    |
| CPU (64G)  | vit_b_lm (LoRA)                | 5                 | 38         | 9:38:11    |
| CPU (64G)  | vit_b_lm (LoRA)                | 10                | 15         | 5:42:34    |


### Plots for the Best Setting:
| Resource   | Model      | Finetuned Strategy | Best Epoch | Train Time |
|------------|------------|--------------------|------------|------------|
| CPU (32G)  | Default    | FFT                | 24         | 5:41:31    |
| CPU (32G)  | Default    | LoRA               | 13         | 2:57:08    |
| CPU (32G)  | Generalist | FFT                | 6          | 2:01:30    |
| CPU (32G)  | Generalist | LoRA               | 7          | 1:58:57    |
| CPU (64G)  | Default    | FFT                | 15         | 3:51:02    |
| CPU (64G)  | Default    | LoRA               | 19         | 5:20:02    |
| CPU (64G)  | Generalist | FFT                | 5          | 1:28:26    |
| CPU (64G)  | Generalist | LoRA               | 15         | 5:42:34    |
| GTX1080    | Default    | MD, PE             | 40         | 1:18:05    |
| GTX1080    | Generalist | MD, PE             | 13         | 0:15:05    |
| RTX5000  | Default      | FFT                | 43         | 0:46:55    |
| RTX5000  | Default      | LoRA               | 16         | 0:17:37    |
| RTX5000  | Generalist   | FFT                | 3          | 0:04:22    |
| RTX5000  | Generalist   | LoRA               | 32         | 0:34:04    |
| V100     | Default      | FFT                | 20         | 0:26:24    |
| V100     | Default      | LoRA               | 42         | 0:51:10    |
| V100     | Generalist   | FFT                | 2          | 0:03:48    |
| V100     | Generalist   | LoRA               | 5          | 0:07:11    |
