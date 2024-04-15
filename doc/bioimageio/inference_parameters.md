# Setting Parameters

## Light Microscopy

### LIVECell

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.6 | 0.7 | 0.5 | 0.5 | 1.2 | 100 |
| vit_b_lm | 0.62 | 0.75 | 0.5 | 0.6 | 1.6 | 100 |
| vit_l_lm | 0.65 | 0.73 | 0.5 | 0.6 | 1.2 | 50 |
| vit_h_lm| 0.65 | 0.73 | 0.5 | 0.5 | 1.6 | 50 |


### DeepBacs

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.6 | 0.75 | 0.6 | 0.6 | 1.8 | 50 |
| vit_b_lm | 0.62 | 0.75 | 0.4 | 0.7 | 2.0 | 100 |
| vit_l_lm | 0.65 | 0.73 | 0.4 | 0.6 | 1.0 | 100 |
| vit_h_lm| 0.65 | 0.73 | 0.4 | 0.6 | 1.0 | 100 |


### TissueNet

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.6 | 0.6 | 0.5 | 0.7 | 1.2 | 50 |
| vit_b_lm | 0.625 | 0.775 | 0.5 | 0.7 |1.8 | 50 |
| vit_l_lm | 0.7 | 0.775 | 0.6 | 0.7 | 1.6 | 50 |
| vit_h_lm| 0.725 | 0.775 | 0.5 | 0.6 | 1.8 | 50 |


### PlantSeg (Root)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.65 | 0.75 | 0.3 | 0.5 | 2.2 | 200 |
| vit_b_lm | 0.68 | 0.85 | 0.3 | 0.6 | 2.2 | 200 |
| vit_l_lm | 0.7 | 0.8 | 0.3 | 0.7 | 2.2 | 200 |
| vit_h_lm| 0.7 | 0.85 | 0.3 | 0.7 | 2.2 | 200 |


### PlantSeg (Ovules)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.6 | 0.68 | 0.5 | 0.5 | 1.4 | 200 |
| vit_b_lm | 0.68 | 0.78 | 0.5 | 0.5 | 1.6 | 200 |
| vit_l_lm | 0.68 | 0.73 | 0.6 | 0.6 | 1.2 | 200 |
| vit_h_lm| 0.7 | 0.73 | 0.7 | 0.6 | 1.4 | 200 |


### NeurIPS CellSeg

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.6 | 0.75 | 0.5 | 0.6 | 2.2 | 100 |
| vit_b_lm | 0.75 | 0.775 | 0.4 | 0.6 | 1.8 | 100 |
| vit_l_lm | 0.775 | 0.8 | 0.6 | 0.6 | 1.4 | 100 |
| vit_h_lm| 0.775 | 0.775 | 0.7 | 0.6 | 2.2 | 100 |    


### Covid IF

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.6 | 0.6 | 0.4 | 0.7 | 2.0 | 200 |
| vit_b_lm | 0.6 | 0.73 | 0.3 | 0.7 | 1.8 | 200 |
| vit_l_lm | 0.62 | 0.7 | 0.4 | 0.5 | 1.2 | 200 |
| vit_h_lm| 0.65 | 0.7 | 0.4 | 0.6 | 1.4 | 200 |    


### DynamicNuclearNet

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.7 | 0.775 | 0.3 | 0.7 | 2.0 | 50 |
| vit_b_lm | 0.8 | 0.825| 0.3 | 0.7 | 2.2 | 50 |
| vit_l_lm | 0.825 | 0.825 | 0.3 | 0.6 | 2.0 | 50 |
| vit_h_lm| 0.85 | 0.85 | 0.5 | 0.7 | 1.8 | 50 |


### HPA

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.65 | 0.7 | 0.4 | 0.3 | 1.4 | 200 |
| vit_b_lm | 0.62 | 0.9| 0.5 | 0.3 | 2.2 | 50 |
| vit_l_lm | 0.88 | 0.88 | 0.3 | 0.3 | 2.2 | 50 |
| vit_h_lm| 0.83 | 0.75 | 0.3 | 0.6 | 1.6 | 200 |


### Lizard

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.62 | 0.8 | 0.4 | 0.4 | 2.0 | 200 |
| vit_b_lm | 0.78 | 0.85 | 0.7 | 0.7 | 1.0 | 200 |
| vit_l_lm | 0.75 | 0.88 | 0.3 | 0.5 | 1.8 | 200 |
| vit_h_lm| 0.68 | 0.88 | 0.7 | 0.7 | 1.0 | 200 |


### PanNuke

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_lm | 0.6 | 0.7 | 0.6 | 0.7 | 1.0 | 200 |
| vit_b_lm | 0.6 | 0.75 | 0.7 | 0.5 | 1.4 | 200 |
| vit_l_lm | 0.6 | 0.775 | 0.6 | 0.5 | 1.6 | 100 |
| vit_h_lm| 0.65 | 0.8 | 0.3 | 0.5 | 1.0 | 50 |


## Electron Micropscopy


### MitoEM (Rat)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.78 | 0.78 | 0.3 | 0.5 | 1.2 | 200 |
| vit_b_em_organelles | 0.78 | 0.83 | 0.3 | 0.4 | 1.4 | 50 |
| vit_l_em_organelles | 0.78 | 0.83 | 0.3 | 0.3 | 1.2 | 200 |
| vit_h_em_organelles| 0.85 | 0.6 | 0.3 | 0.4 | 1.2 | 200 |


### MitoEM (Human)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.8 | 0.75 | 0.3 | 0.3 | 1.2 | 50 |
| vit_b_em_organelles | 0.78 | 0.83 | 0.3 | 0.3 | 1.2 | 200 |
| vit_l_em_organelles | 0.8 | 0.85 | 0.3 | 0.3 | 1.6 | 100 |
| vit_h_em_organelles| 0.8 | 0.85 | 0.3 | 0.3 | 1.2 | 50 |


### Platynereis (Nuclei)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.85 | 0.875 | 0.3 | 0.7 | 1.8 | 200 |
| vit_b_em_organelles | 0.825 | 0.85 | 0.3 | 0.7 | 2.0 | 200 |
| vit_l_em_organelles | 0.9 | 0.9 | 0.3 | 0.7 | 2.0 | 200 |
| vit_h_em_organelles| 0.85 | 0.825 | 0.3 | 0.7 | 2.0 | 200 |


### Lucchi

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.75 | 0.75 | 0.3 | 0.6 | 2.2 | 200 |
| vit_b_em_organelles | 0.8 | 0.83 | 0.3 | 0.4 | 2.2 | 200 |
| vit_l_em_organelles | 0.83 | 0.83 | 0.3 | 0.5 | 2.2 | 200 |
| vit_h_em_organelles| 0.83 | 0.75 | 0.3 | 0.4 | 1.8 | 200 |


### UroCell

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.675 | 0.85 | 0.5 | 0.7 | 1.0 | 100 |
| vit_b_em_organelles | 0.725 | 0.85 | 0.3 | 0.7 | 1.4 | 100 |
| vit_l_em_organelles | 0.825 | 0.85 | 0.3 | 0.7 | 1.6 | 100 |
| vit_h_em_organelles| 0.825 | 0.85 | 0.3 | 0.6 | 1.2 | 50 |


### VNC

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.625 | 0.75 | 0.5 | 0.3 | 1.8 | 200 |
| vit_b_em_organelles | 0.775 | 0.8 | 0.5 | 0.3 | 1.6 | 200 |
| vit_l_em_organelles | 0.7 | 0.825 | 0.3 | 0.6 | 1.0 | 50 |
| vit_h_em_organelles| 0.75 | 0.8 | 0.4 | 0.7 | 1.0 | 200 |


### ASEM (Mito)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.625 | 0.8 | 0.3 | 0.5 | 1.0 | 200 |
| vit_b_em_organelles | 0.6 | 0.9 | 0.6 | 0.7 | 1.6 | 200 |
| vit_l_em_organelles | 0.925 | 0.85 | 0.3 | 0.7 | 1.0 | 200 |
| vit_h_em_organelles| 0.85 | 0.6 | 0.3 | 0.7 | 1.0 | 50 |


### MitoLab (C. elegans)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.78 | 0.78 | 0.6 | 0.7 | 2.0 | 50 |
| vit_b_em_organelles | 0.78 | 0.78 | 0.3 | 0.4 | 1.0 | 50 |
| vit_l_em_organelles | 0.78 | 0.75 | 0.3 | 0.7 | 1.0 | 50 |
| vit_h_em_organelles| 0.8 | 0.75 | 0.7 | 0.7 | 1.6 | 50 |


### MitoLab (Fly Brain)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.6 | 0.85 | 0.3 | 0.3 | 1.0 | 50 |
| vit_b_em_organelles | 0.83 | 0.6 | 0.5 | 0.5 | 1.0 | 100 |
| vit_l_em_organelles | 0.83 | 0.83 | 0.3 | 0.4 | 1.8 | 50 |
| vit_h_em_organelles| 0.83 | 0.6 | 0.3 | 0.6 | 1.8 | 50 |


### MitoLab (Glycotic Muscle)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.6 | 0.65 | 0.4 | 0.4 | 1.0 | 100 |
| vit_b_em_organelles | 0.65 | 0.73 | 0.4 | 0.5 | 1.8 | 100 |
| vit_l_em_organelles | 0.62 | 0.7 | 0.4 | 0.6 | 1.0 | 50 |
| vit_h_em_organelles| 0.83 | 0.7 | 0.3 | 0.6 | 1.0 | 100 |


### MitoLab (HeLa Cell)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.78 | 0.6 | 0.5 | 0.7 | 1.0 | 100 |
| vit_b_em_organelles | 0.78 | 0.8 | 0.3 | 0.3 | 1.6 | 50 |
| vit_l_em_organelles | 0.8 | 0.78 | 0.3 | 0.3 | 1.0 | 50 |
| vit_h_em_organelles| 0.78 | 0.83 | 0.3 | 0.3 | 1.4 | 100 |


### MitoLab (TEM)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.75 | 0.75 | 0.6 | 0.7 | 1.4 | 200 |
| vit_b_em_organelles | 0.725 | 0.8 | 0.5 | 0.5 | 1.8 | 100 |
| vit_l_em_organelles | 0.8 | 0.8 | 0.4 | 0.5 | 1.8 | 100 |
| vit_h_em_organelles| 0.8 | 0.75 | 0.4 | 0.6 | 1.8 | 50 |


### NucMM (Mouse)

| model | iou_threshold | stability_score_thresh | *center_distance_threshold* | *boundary_distance_threshold* | *distance_smoothing* | *min_size*
|--|--|--|--|--|--|--|
| vit_t_em_organelles | 0.65 | 0.65 | 0.6 | 0.6 | 1.0 | 50 |
| vit_b_em_organelles | 0.68 | 0.75 | 0.7 | 0.4 | 1.0 | 50 |
| vit_l_em_organelles | 0.62 | 0.73 | 0.5 | 0.4 | 1.2 | 50 |
| vit_h_em_organelles| 0.68 | 0.7 | 0.6 | 0.4 | 1.4 | 50 |
