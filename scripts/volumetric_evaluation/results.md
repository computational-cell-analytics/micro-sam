# Results: Volumetric 3d Segmentation

## Lucchi

### Interactive Instance Segmentation

| model               | iou_threshold | projection   | box_extension | mSA                | SA50               |
|---------------------|---------------|--------------|---------------|--------------------|--------------------|
| vit_b               |     0.8       |     box      |     0.0       | 0.3479752912354858 | 0.6744186046511628 |
| vit_b_em_organelles |     0.6       | single_point |     0.0       | 0.4740420971988575 | 0.7560975609756098 |


### Automatic Instance Segmentation*

- For `vit_b_em_organelles`:
```
mSA = 0.24586712213916712
SA50 = 0.45454545454545453
```


## MitoEM (Rat)

### Interactive Instance Segmentation

| model               | iou_threshold | projection   | box_extension | mSA                | SA50               |
|---------------------|---------------|--------------|---------------|--------------------|--------------------|
| vit_b               |     0.5       | single_point |     0.025     | 0.4460985168927346 | 0.7087378640776699 |
| vit_b_em_organelles |     0.5       | points       |     0.05      | 0.5515894109242193 | 0.8333333333333334 |


### Automatic Instance Segmentation*

- For `vit_b_em_organelles`:
```
mSA = 0.4817079936297749
SA50 = 0.7064220183486238
```


## MitoEM (Human)

### Interactive Instance Segmentation

| model               | iou_threshold | projection      | box_extension | mSA                | SA50               |
|---------------------|---------------|-----------------|---------------|--------------------|--------------------|
| vit_b               |     0.7       | points_and_mask |     0.0       | 0.2570121772955602 | 0.5642458100558659 |
| vit_b_em_organelles |     0.6       | points          |     0.075     | 0.4345798086356913 | 0.75               |


### Automatic Instance Segmentation*

- For `vit_b_em_organelles`:
```
mSA = 0.34706033223263305
SA50 = 0.6057142857142858
```


## PlantSeg (Ovules)

### Interactive Instance Segmentation

| model    | iou_threshold | projection   | box_extension | mSA                | SA50                |
|----------|---------------|--------------|---------------|--------------------|---------------------|
| vit_b    |     0.8       | box          |     0.0       | 0.1298936240680811 | 0.382522671063479   |
| vit_b_lm |     0.7       | single_point |     0.0       | 0.1923181620045619 | 0.45281385281385284 |


### Automatic Instance Segmentation*

- For `vit_b_lm`:
```
mSA = 0.114352715111846
SA50 = 0.2702702702702703
```


## PlantSeg (Root)

### Interactive Instance Segmentation

| model    | iou_threshold | projection | box_extension | mSA                | SA50                |
|----------|---------------|------------|---------------|--------------------|---------------------|
| vit_b    |     0.8       | box        |     0.0       | 0.0775321999839266 | 0.225               |
| vit_b_lm |     0.8       | box        |     0.025     | 0.1861581171532632 | 0.36904761904761907 |


### Automatic Instance Segmentation*

- For `vit_b_lm`:
```
mSA = 0.08115580115011829
SA50 = 0.16464891041162227
```


> NOTE: *We use the same parameters for automatic instance segmentation -
`center_distance_threshold` = 0.3, `boundary_distance_threshold` = 0.4, `distance_smoothing` = 2.2, `min_size` = 200, `gap_closing` = 2, `min_z_extent` = 2