# Dice-foreground AIS decoder re-optimization

This is the bounded follow-up comparison of the already trained `baseline.pt` and `boundary.pt` checkpoints.
It does not train another decoder and does not include a foreground-BCE variant. Both checkpoints were trained
with the Dice foreground objective; the boundary model additionally predicts the full object-boundary channel.
Here “Dice” distinguishes these checkpoints from the foreground-calibration (`fgcal`) experiments: the already
trained auxiliary boundary head retains the Dice-plus-BCE loss recorded by its checkpoint's training code.

## Fixed protocol

- Checkpoints: `<staged>/baseline.pt` and `<staged>/boundary.pt`, where `<staged>` is
  `/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/ais_decoder_training/staged/joint_sam2_hvit_t_multi_gpu`.
- Development datasets (equal weight): LIVECell, TissueNet, DynamicNuclearNet, DeepBacs, YeaZ,
  NeurIPS CellSeg, PUMA, TNBC and COVID-IF. These are `primary training_extra` with DIC-HepG2 and DeepSeas
  excluded. No 3D data enters selection.
- Diagnostic holdout: LIVECell, TissueNet and DynamicNuclearNet only. Reused DeepBacs is excluded. This is a
  robustness check, not another selection stage.
- Sealed OOD confirmation: Arvidsson 10/10, BitDepth NucSeg 70/70 (reported equally over four magnifications),
  CellBinDB 48 (8 per six stain/acquisition types), microbeSEG 2/2 manual test images and VICAR 50 (10 per five
  cell types). Inputs are centre-cropped to at most 512 x 512; smaller images remain at native size.
- The baseline and boundary checkpoint each get their own configuration. A shared post-processing setting is not
  used for the checkpoint comparison.

`benchmark_apg_optimization.py --prepare-only --subset ood_extended --ndim 2` creates and freezes the OOD
manifest as `subset_manifest_v5_ood_extended.json`. Selection uses the test loaders, validates both raw and label
files, and refuses changed counts or strata. `report_ais_checkpoint_comparison.py` independently checks its paths
against both decoder `data_manifest.json` files before it reports a result.

## Search

The corrected cached scorer in `parameter_search.py` now uses channel 4 exactly like production AIS:
`contact_weight` raises the watershed height at boundaries; `contact_mask_threshold` runs the open-mask watershed
and re-flood. These `contact_*` spellings are legacy post-processing API names; for `boundary.pt`, channel 4 is
the full object-boundary probability, not the earlier touching-contact target. The scorer rejects those
parameters for a four-channel checkpoint. JSON `null` consistently means “use the model default”; only the
explicit string `"off"` disables the default boundary-magnitude filter.

Coarse candidate families:

- baseline: `configs/ais_dice_reopt_base.json` (1,344 combinations);
- boundary: `configs/ais_dice_reopt_boundary.json` contains the no-auxiliary (1,344), ridge (4,032), mask
  (4,032), and ridge-plus-mask (1,344) families in one 10,752-candidate sweep. The four component JSON files
  retain the individual family specifications for inspection.

The grids cover foreground threshold 0.30-0.60, density threshold 5-50, size 25/50, sigma 0.5/1.0,
400-1,600 flow iterations and foreground height weight 0.5/0.75/1.0. The boundary families jointly search these
with ridge weight or mask threshold. The second stage is deliberately local: take the top three rows of each
mechanism family and vary one coordinate. `prepare_ais_reoptimization_polish.py` creates this explicit candidate
grid and only extends flow to 2,400 iterations when the 1,600-iteration edge still gains at least 0.001 mSA (and
similarly tests 200 only when the 400 edge beats 800).

Sweep sharding is cache-aware: every configuration with the same foreground threshold, smoothing, iteration
count and step size stays in one shard. Thus the expensive flow density is computed once per image and flow group,
not once per shard. The consolidated boundary grid also shares that computation across all four mechanism
families. Shards still form an exact disjoint partition of the requested combinations. The `cpu-test` submission
preset packs up to 48 four-thread shard commands into one exclusive 192-core test-node allocation. The coarse
layout uses 12 shards for each of four primary datasets (48 commands) and 9 for each of five training-extra
datasets (45 commands), so every submitted array occupies one node and uses most of its cores.

Rank with `report_ais_sweep.py --no-reference`. It unions multiple mechanism grids and selects from the rows no
more than 0.001 mSA below the best. Within that plateau it favours the best worst-dataset relative optimum, then
fewer flow iterations and fewer boundary controls. The emitted JSON is directly accepted by `run`.

## Execution recipe

From `finetuning/v2/evaluation/optimization`, with the `new-stack` environment active:

```bash
export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/ais_decoder_training/staged
export MICRO_SAM2_JOINT_EXPORT_ROOT=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/model_exports
ROOT=/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization
REP=$ROOT/ais/reports/dice_reoptimization
PRIMARY_CORE="livecell tissuenet dynamicnuclearnet deepbacs"
EXTRA_CORE="yeaz neurips_cellseg puma tnbc covid_if"
CORE="$PRIMARY_CORE $EXTRA_CORE"
```

Prepare the manifest and cache development predictions. The four `--print-only` invocations print the task
graphs to inspect; omit that flag only after checking them.

```bash
python benchmark_apg_optimization.py --prepare-only --subset ood_extended --ndim 2
python ais_campaign_tasks.py predict --name dice_base_primary_predict --subsets primary --print-only \
  --extra "--joint-checkpoint baseline --ndim 2 --datasets $PRIMARY_CORE"
python ais_campaign_tasks.py predict --name dice_base_extra_predict --subsets training_extra --print-only \
  --extra "--joint-checkpoint baseline --ndim 2 --datasets $EXTRA_CORE"
python ais_campaign_tasks.py predict --name dice_boundary_primary_predict --subsets primary --print-only \
  --extra "--joint-checkpoint boundary --ndim 2 --datasets $PRIMARY_CORE"
python ais_campaign_tasks.py predict --name dice_boundary_extra_predict --subsets training_extra --print-only \
  --extra "--joint-checkpoint boundary --ndim 2 --datasets $EXTRA_CORE"
```

Run the coarse sweeps. Primary and training-extra are separate because their dataset sets do not overlap. Use
distinct `--name` values for each checkpoint and subset. The 12/9 shard counts pack each array into one full
test node and do not change the candidate set.

```bash
python ais_campaign_tasks.py sweep --name dice_base_primary --preset cpu-test --subsets primary \
  --grid configs/ais_dice_reopt_base.json --datasets $PRIMARY_CORE --num-shards 12 --print-only \
  --extra "--joint-checkpoint baseline --ndim 2 --mode sparse"
python ais_campaign_tasks.py sweep --name dice_base_extra --preset cpu-test --subsets training_extra \
  --grid configs/ais_dice_reopt_base.json --datasets $EXTRA_CORE --num-shards 9 --print-only \
  --extra "--joint-checkpoint baseline --ndim 2 --mode sparse"
python ais_campaign_tasks.py sweep --name dice_boundary_primary --preset cpu-test --subsets primary \
  --grid configs/ais_dice_reopt_boundary.json --datasets $PRIMARY_CORE --num-shards 12 --print-only \
  --extra "--joint-checkpoint boundary --ndim 2 --mode sparse"
python ais_campaign_tasks.py sweep --name dice_boundary_extra --preset cpu-test --subsets training_extra \
  --grid configs/ais_dice_reopt_boundary.json --datasets $EXTRA_CORE --num-shards 9 --print-only \
  --extra "--joint-checkpoint boundary --ndim 2 --mode sparse"
```

After every shard succeeds, merge each grid with one `benchmark_ais_optimization.py sweep --merge` call using
`--num-shards 12` for primary and `--num-shards 9` for training_extra, then rank:

```bash
mkdir -p "$REP"
python report_ais_sweep.py --grid configs/ais_dice_reopt_base.json --subset primary training_extra \
  --datasets $CORE --joint-checkpoint baseline --no-reference --output "$REP/baseline_coarse.csv"
python report_ais_sweep.py --grid configs/ais_dice_reopt_boundary.json \
  --subset primary training_extra --datasets $CORE \
  --joint-checkpoint boundary --no-reference --output "$REP/boundary_coarse.csv"
python prepare_ais_reoptimization_polish.py --ranking "$REP/baseline_coarse.csv" \
  --output configs/ais_dice_reopt_baseline_polish.json
python prepare_ais_reoptimization_polish.py --ranking "$REP/boundary_coarse.csv" \
  --output configs/ais_dice_reopt_boundary_polish.json
```

Each generator command prints the number of distinct flow-cache groups. Set `--num-shards` no higher than that
printed count. A polish grid is much smaller than the coarse grid: submit its individual shard commands with
`cpu-shared`, or combine the primary and training-extra task lists before using the packed `cpu-test` preset.
The sweep rejects a larger shard count instead of producing empty result files. Then merge with the same chosen
count, rank the union and write the two own-optimum configs:

```bash
python report_ais_sweep.py --grid configs/ais_dice_reopt_base.json \
  configs/ais_dice_reopt_baseline_polish.json --subset primary training_extra --datasets $CORE \
  --joint-checkpoint baseline --no-reference --output "$REP/baseline_final.csv" \
  --select-config "$REP/baseline_optimum.json" --config-name baseline-dice-optimum
python report_ais_sweep.py --grid configs/ais_dice_reopt_boundary.json \
  configs/ais_dice_reopt_boundary_polish.json \
  --subset primary training_extra --datasets $CORE --joint-checkpoint boundary --no-reference \
  --output "$REP/boundary_final.csv" --select-config "$REP/boundary_optimum.json" \
  --config-name boundary-dice-optimum
```

Run the own-optimum configs on the three-dataset disjoint holdout for diagnosis. Only after configs are frozen,
cache and score `ood_extended` for both checkpoints. Keep diagnostics enabled. Generate these task graphs with:

```bash
HOLDOUT="livecell tissuenet dynamicnuclearnet"
python ais_campaign_tasks.py screen --name dice_base_holdout --preset cpu-shared --subsets holdout --no-defaults \
  --configs "$REP/baseline_optimum.json" --print-only \
  --extra "--joint-checkpoint baseline --ndim 2 --datasets $HOLDOUT"
python ais_campaign_tasks.py screen --name dice_boundary_holdout --preset cpu-shared --subsets holdout --no-defaults \
  --configs "$REP/boundary_optimum.json" --print-only \
  --extra "--joint-checkpoint boundary --ndim 2 --datasets $HOLDOUT"
python ais_campaign_tasks.py predict --name dice_base_ood --subsets ood_extended --print-only \
  --extra "--joint-checkpoint baseline --ndim 2"
python ais_campaign_tasks.py predict --name dice_boundary_ood --subsets ood_extended --print-only \
  --extra "--joint-checkpoint boundary --ndim 2"
python ais_campaign_tasks.py screen --name dice_base_ood_score --preset cpu-shared --subsets ood_extended \
  --no-defaults --configs "$REP/baseline_optimum.json" --print-only \
  --extra "--joint-checkpoint baseline --ndim 2"
python ais_campaign_tasks.py screen --name dice_boundary_ood_score --preset cpu-shared --subsets ood_extended \
  --no-defaults --configs "$REP/boundary_optimum.json" --print-only \
  --extra "--joint-checkpoint boundary --ndim 2"
```

Submit the score tasks only after their prediction tasks finish. Finally pass the two OOD run directories and
the frozen manifest to:

```bash
python report_ais_checkpoint_comparison.py --baseline-runs <baseline-ood-run> \
  --boundary-runs <boundary-ood-run> --manifest "$ROOT/subset_manifest_v5_ood_extended.json" \
  --output "$REP/ood_confirmation.json"
```

## Decision rule

The JSON report supports a strong boundary-decoder improvement statement only when all four conditions hold:

1. the 95% paired hierarchical-bootstrap CI for the macro mSA difference is above zero;
2. at least four of the five OOD domains improve;
3. no domain loses both more than 0.005 absolute mSA and more than 2% relative;
4. equal-domain macro mSA improves by at least 2% relative.

The bootstrap resamples domains, then paired source images within every domain/acquisition stratum. The report
also writes per-domain and paired-sample CSVs, object-fate diagnostic deltas, generation time, both selected
parameter dictionaries, checkpoint/implementation checksums and the training-disjointness audit. microbeSEG is
always labelled as an `n=2` stress test, not treated as precise standalone evidence.
