# Plotting Scripts

- `plot_em_organelles.py` - The script for plotting all the results* from Electron Microscopy Mitochondria - Nuclei datasets (MitoNet, vanilla Segment Anything models, generalist models)

- `plot_livecell.py` - The script for plotting all the results* from LiveCELL dataset (CellPose, vanilla Segment Anything models, specialist models, generalist models)

- `plot_lm.py` - The script for plotting all the results* from Light Microscopy datasets (CellPose, vanilla Segment Anything models, generalists)

- `plot_train_grid_search.py` - The script for plotting grid search experiment results* from LiveCELL specialists.

- `plot_explorative_experimets.py` - The scripts for plotting several experiments (partial finetuning, n_objects_per_batch for iterative training loop)


> *results: `amg` (automatic mask generation) ,`ais` (automatic instance segmentation), `1pn0` (1 positive point & 0 negative point prompt), `box` (box prompt), i<sub>p</sub> (last iteration of iterative prompting starting with point), i<sub>b</sub> (last iteration of iterative prompting starting with box), 
