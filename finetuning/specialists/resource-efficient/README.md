Questions:

a. Resource combinations: (see `run_resource_efficient_finetuning.py`)

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
