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

| dataset | shape (Z x Y x X) | mSA | SA50 | VI-split | VI-merge | CREMI | time (total) |
|---------|-------------------|-----|------|----------|----------|-------|--------------|
| nis3d | 198 x 978 x 987 | 0.4562 | 0.8069 | - | - | - | 5m48s |
| plantseg_ovules | 320 x 960 x 1000 | 0.3771 | 0.5941 | - | - | - | 12m33s |
| liconn (EM protocol) | 700 x 700 x 700 | 0.0004 | - | 8.2262 | 1.4307 | 3.1045 | 11m28s |
| mitoem-human (EM) | 100 x 4096 x 4096 | 0.0000 | - | 9.7326 | 0.2464 | 3.0998 | 46m14s |
| cremi-padded sampleB (EM) | 200 x 3072 x 3072 | 0.0180 | - | 2.0314 | 1.0036 | 0.9379 | 46m44s |
| microns minnie65 (EM) | 200 x 2048 x 2048 | 0.0015 | - | 3.4523 | 2.4439 | 1.6291 | 28m18s |
