Questions:

a. Resource combinations:
    - `xps13` (CPU compute) (local)
    - `medium` (CPU compute partition) (SCC)
    - `gtx1080`: 8GB (SCC)
    - `rtx5000`: 16GB (SCC)
    - `v100`: 32GB (SCC / Grete)
    - `A100`: 40GB (Grete)

b. Experiment combinations:
    i. `vit_t` / `vit_b` / both
    ii. number of training images (1/2/5/10)
    iii. number of validation images (set to 1)
    iv. number of objects per batch (X to fit all)
    v. number of epochs for different number of training samples
        - idea: let's keep the number of epochs per combination training to stay the same (for consistency in early stopping, lr scheduler)

LP:
> freeze parts of SAM?
> learning rate (starting, scheduler choice), optimizer

c. Inference:
    i. vanilla SAM
    ii. `vit_<X>_lm` micro-sam (using finetuned LM generalist)
    iii. `vit_<X>_covid-if` micro-sam (training a specialist)
    iv. finetuning `vit_<X>_lm` microsam (finetuning the LM generalist)
