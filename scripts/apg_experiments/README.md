## Experiments for Automatic Prompt Generation (APG)

This folder contains evaluaton code for applying the new APG method, built on `micro-sam` to microscopy data using the `micro_sam` library. This code was used for our experiments in preparation of the [manuscript](https://openreview.net/forum?id=xFO3DFZN45).

Please note that this folder may become outdated due to changes in function signatures, etc., and often does not use the functionality that we recommend to users. We also will not actively maintain the code here. Please refer to the [example notebooks](https://github.com/computational-cell-analytics/micro-sam/tree/master/notebooks) and [example scripts](https://github.com/computational-cell-analytics/micro-sam/tree/master/examples) for well maintained and documented `micro-sam`'s APG examples.

### Evaluation Scripts:

The top-level folder contains scripts to evaluate other models with `micro-sam`, and the `plotting` subfolder contains scripts for visualizations and plots for the manuscript.

- `analyze_posthoc.py`: Experiments to visually understand and debug the APG method.
- `perform_posthoc.py`: Experiments to store results to work with `analyze_posthoc.py`.
- `prepare_baselines.py`: Experiments to run all methods presented in the manuscript with default parameters on all microscopy imaging data.
- `run_evaluation.py`: Experiments related to APG for understanding the hyperparameters using grid-search.
- `submit_evaluation.py`: Scripts for submitting jobs to slurm.
- `util,py`: Scripts containing data processing scripts and other miscellanous stuff.
- `plotting`/
    - `calculate_mean.py`: Scripts to calculate mean performance metric over per modality per method.
    - `plot_ablation.py`: Scripts to show results for comparing connected components- vs. boundary distance-based prompt extraction.
    - `plot_average.py`: Same as `calculate_mean.py`. The scripts plot the mean values in a barplot and displays absolute rank per method over all datasets per modality.
    - `plot_qualitative.py`: Scripts to display qualitative results over all datasets.
    - `plot_quantitative.py`: Scripts to display quantitative results over all datasets.
    - `plot_util.py`: Stores related information helpful for plotting.
- `statistical_analysis`: Scripts for performing statistical analysis on quantitative results computed per image.
