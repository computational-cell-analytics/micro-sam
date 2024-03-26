# Results: Volumetric 3d Segmentation

## Lucchi

### Interactive Instance Segmentation

| model              | iou_threshold | projection   | box_extension | mSA                |
|--------------------|---------------|--------------|---------------|--------------------|
| vit_b              |     0.8       |     box      |     0.0       | 0.3479752912354858 |
| vit_b_em_organelles |     0.6       | single_point |     0.0       | 0.4740420971988575 |

## MitoEM (Rat)

### Interactive Instance Segmentation

| model              | iou_threshold | projection   | box_extension | mSA                |
|--------------------|---------------|--------------|---------------|--------------------|
| vit_b              |     0.5       | single_point |     0.025     | 0.4460985168927346 |
| vit_b_em_organelles |     0.5       | points       |     0.05      | 0.5515894109242193 |

## MitoEM (Human)

### Interactive Instance Segmentation

| model              | iou_threshold | projection      | box_extension | mSA                |
|--------------------|---------------|-----------------|---------------|--------------------|
| vit_b              |     0.7       | points_and_mask |     0.0       | 0.2570121772955602 |
| vit_b_em_organelles |     0.6       | points          |     0.075     | 0.4345798086356913 |

## PlantSeg (Ovules)

### Interactive Instance Segmentation

| model   | iou_threshold | projection   | box_extension | mSA                |
|---------|---------------|--------------|---------------|--------------------|
| vit_b   |     0.8       | box          |     0.0       | 0.1298936240680811 |
| vit_b_lm |     0.7       | single_point |     0.0       | 0.1923181620045619 |

## PlantSeg (Root)

### Interactive Instance Segmentation

| model   | iou_threshold | projection | box_extension | mSA                |
|---------|---------------|------------|---------------|--------------------|
| vit_b   |     0.8       | box        |     0.0       | 0.0775321999839266 |
| vit_b_lm |     0.8       | box        |     0.025     | 0.1861581171532632 |
