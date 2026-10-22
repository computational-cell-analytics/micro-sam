# LICONN demo

Interactive and automatic segmentation of a LICONN block with micro-sam2 (`hvit_l_cells`).

Precompute once on a GPU:

```bash
python run_interactive_segmentation.py --precompute
python run_automatic_segmentation.py --precompute ais
python run_automatic_segmentation.py --precompute apg
```

Then open napari (no GPU needed):

```bash
python run_interactive_segmentation.py  # annotator
python run_automatic_segmentation.py  # AIS and APG results
```

The default paths are on the GWDG cluster. Elsewhere, add `--data_root /path/to/liconn --demo_root /path/to/output` to each command.
If the data is missing, the scripts download the full LICONN volume to `--data_root`.
