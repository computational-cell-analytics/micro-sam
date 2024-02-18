Questions:

a. Resource combinations: (see `run_resource_efficient_finetuning.py`)
b. Experiment combinations:
    i. `vit_t` / `vit_b` / both
    ii. number of training samples (1/2/5/10)
    iii. number of validation samples?
        - we keep the number of validation samples consistent (eg. 5)
        - we keep the number of validation samples to 1
    iv. number of objects per batch (25 is difficult to fit on `gtx1080`)
    v. freeze parts of SAM?
    vi. patch shape?
    vii. number of epochs for different number of training samples
        - idea: let's keep the number of epochs per combination training to stay the same (for consistency in early stopping, lr scheduler)
    viii. learning rate (starting, scheduler choice), optimizer
