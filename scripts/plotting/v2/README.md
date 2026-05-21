# plotting/v2

Scripts for postprocessing experiments and visualization of UniSAM2 predictions.

## Scripts

- **`run_predictions.py`** — Run UniSAM2 model inference on a dataset and cache raw/distances/labels
  to H5 files for offline use (e.g. bulk re-runs without GPU).
- **`grid_search_postprocessing.py`** — Sweep postprocessing hyperparameters (flow thresholds, sigma,
  etc.) using the cpp backend. Runs inference live; results saved to `OUTPUT_ROOT`.
- **`visualize.py`** — Load cached H5/TIF predictions and open them in napari for inspection.

## Workflow

```bash
# 1. Grid search (runs model live, sweeps postprocessing params, saves CSV)
python grid_search_postprocessing.py -d nis3d -m automatic
python grid_search_postprocessing.py -d plantseg_ovules -m automatic
python grid_search_postprocessing.py -d humanneurons -m automatic

# 2. Full-volume prediction with best params (no crop, saves H5)
python grid_search_postprocessing.py -d nis3d -m automatic --full_volume
python grid_search_postprocessing.py -d plantseg_ovules -m automatic --full_volume
python grid_search_postprocessing.py -d humanneurons -m automatic --full_volume

# 3. Visualize in napari
python visualize.py --dataset nis3d
```

## Output

Grid search CSVs and full-volume H5s are written to:

    /mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/grid-search-experiments/

## Results (unisam2-automatic, cpp backend)

Full-volume evaluation on the first test sample per dataset (no crop).
mSA: higher is better. VI-split / VI-merge: lower is better (EM only).

| dataset | shape (Z x Y x X) | mSA | SA50 | VI-split | VI-merge | CREMI |
|---------|-------------------|-----|------|----------|----------|-------|
| nis3d | 198 x 978 x 987 | 0.4562 | 0.8069 | - | - | - |
| plantseg_ovules | 320 x 960 x 1000 | 0.3771 | 0.5941 | - | - | - |
| humanneurons (EM) | 64 x 2048 x 2048 | 0.0179 | 0.0505 | 3.3183 | 1.1899 | 2.0375 |
| cremi (EM) | 125 x 1250 x 1250 | 0.0081 | 0.0135 | 1.0537 | 0.4314 | 0.5728 |
| snemi (EM) | 100 x 1024 x 1024 | 0.0608 | 0.1189 | 2.2171 | 0.7810 | 1.3634 |
