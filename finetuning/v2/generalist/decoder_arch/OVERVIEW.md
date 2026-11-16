# Decoder architecture evaluation: overview (status 2026-11-16)

Which segmentation decoder should UniSAM2 (`hvit_t`) use for automatic instance segmentation (AIS) in 2D and 3D?
The detailed plan is `/user/pape41/u12086/.claude/plans/decoder-architecture-evaluation.md`. `README.md` in this
folder documents the workflow, the evaluation protocol and the deviations from the plan. Checkpoints, job scripts,
logs and results are under `/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_arch`.

## Step 1 arms

| Arm | Decoder | Decoder params |
|---|---|---|
| A | current UniSAM2 decoder (no real skip connections; reference) | 6.40 M |
| B | A with real stride-4 / stride-8 skip features and an image stem | 5.58 M |
| C | Hiera-U: skip U-decoder, 2D convs on folded slices, zero-init z branches | 1.56 M |

Every arm trains identically apart from the decoder:
- **Model and loss:** SAM2.1 `hvit_t` encoder, production v6 targets and loss.
- **Data:** a fixed train / validation manifest. The PlantSeg root train volume that duplicates a test volume
  is removed.
- **Schedule:** 31,800 iterations (53 epochs, calibrated to fit 8 h for the slowest arm, A), decoder learning
  rate 1e-4, encoder learning rate 1e-5.
- **Convergence:** all runs had converged. Over the last 20 % of training, the validation loss improved by
  at most 0.4 %.

## Implementation

Library:
- `micro_sam/v2/models/decoders.py`: decoders B, C (step 1), D and E (step 2).
- `UniSAM2(decoder_type=...)` builds the decoder. Checkpoints store `decoder_config`, from which
  `get_unisam2_model` rebuilds the decoder.
- Tests: `test/test_models/test_unisam2_decoders.py`.

Scripts in this folder:

| Script | Purpose |
|---|---|
| `build_manifest.py` | fixed train / validation subsets and the leakage check |
| `train_decoder_arch.py`, `decoder_arch_lib.py` | training of one arm, speed calibration |
| `summarize_calibration.py` | common iteration count from the calibration |
| `submit_decoder_arch.py` | SLURM submission (`calibrate`, `train`, `evaluate`), from a frozen copy of the code |
| `evaluate_decoder_arch.py`, `eval_lib.py` | evaluation of one run: `predict-val` → `tune` → `test` |
| `report_decoder_arch.py` | comparison of all runs: tables, paired bootstrap, diagnostics |
| `test_eval_lib.py` | unit tests of the evaluation |

How a run is evaluated:
- **Tuning:** the flow post-processing is tuned on the validation data, one parameter set for 2D and one for
  3D, by grid search.
- **Testing:** 15 test sets are scored, 12 of them reported (6 in-domain, 6 external), with the run's own
  tuned parameters and with the library defaults.
- **Scoring:** test scoring reproduces the existing evaluation harness (`finetuning/v2/evaluation/common.py`)
  exactly.
- **Diagnostics:** foreground IoU, boundary recall, merged / split / missed objects, and the decoder's time
  and memory.

## Results

Full tables: `<experiment root>/evaluation/report/report.md`. mSA on the test sets, "headline" = mean over
the 12 reported datasets.

| Run | Headline (tuned) | 2D | 3D | Headline (defaults) | Decode Z=8 | Peak memory Z=8 |
|---|---|---|---|---|---|---|
| A, seed 0 | 0.292 | 0.365 | 0.219 | 0.280 | 232 ms | 8.9 GiB |
| A, seed 1 | 0.297 | 0.365 | 0.230 | 0.302 | 232 ms | 8.9 GiB |
| B, seed 0 | 0.308 | 0.372 | 0.243 | 0.315 | 198 ms | 9.4 GiB |
| C, seed 0 | **0.339** | **0.384** | **0.294** | **0.328** | **79 ms** | **4.5 GiB** |

Decode times and memory are for 8 slices of 1024 px on an A100.

- **C is clearly best.** Against A seed 0 it gains +0.048 on the headline (+16 %), +5 % in 2D and +35 % in 3D.
  The gain holds at default parameters and on the external test sets (2D +9 %, 3D +25 %). It is far larger
  than the difference between the two A seeds (0.006). C is also the cheapest decoder: a quarter of A's
  parameters, a third of the decode time, half the memory, and about half the training time (3.4 h vs 6.7 h).
- **C fixes the 3D merges.** The current decoder merges many 3D objects; for example, 51 % of NIS3D nuclei are
  merged with a neighbour. C reduces this to 13 % on NIS3D and improves CShaper by 75 % and PlantSeg ovules by
  62 %.
- **C's weak spot is GoNuclear:** −13 % with tuned parameters, but within 1 % at default parameters. The loss
  comes from C's tuned 3D parameters, not from the decoder (see the 3D tuning caveat below).
- **B, skip connections alone, helps less.** It gains +0.016 on the headline, mostly on the in-domain 3D sets.
  It is 8 % worse on the external 3D sets and loses 34 % on CShaper (21 % at defaults).

### Caveats

- **3D tuning does not transfer well.** The 3D parameters tuned on the 18 validation crops often score worse
  on the test sets than the defaults. The two A seeds also select very different 3D parameters. Tuning
  probably adds noise to 3D results until this is fixed.
- **Seed noise per dataset is about 5–7 %.** The two A seeds agree on the headline (0.006) and in 2D (0.0003),
  but individual datasets differ by up to 5–7 % (CellPose, EmbedSeg, Usiigaci), and PlantSeg ovules by +53 %.
- **Some test sets are small.** NIS3D and three EmbedSeg collections have 2 test crops each, and GoNuclear is
  a single volume. Usiigaci contains 37 time points of one field of view.

### On the decision rule

- **The plan's rule is not suitable.** Its per-dataset check ("no dataset loses more than 5 % with a 95 %
  interval below zero") uses bootstrap intervals over test images. Those intervals do not capture the
  variation between training seeds.
- **Evidence:** the seed replicate A seed 1 would itself fail the check on three datasets.
- **Formal verdicts:** under the rule, B and C are "does not beat the reference" (B because of CShaper,
  C because of GoNuclear).
- **The report still prints these verdicts, but they are not used.** Decisions are made from the results
  directly, and the step 1 results are clear: C is the best decoder.

## Step 2

| Arm | Decoder | Decoder params |
|---|---|---|
| D | CAGD-3D (`cagd_3d`): MedCAGD-style context-aggregating decoder, hybrid 2D/3D, deep supervision | 8.51 M |
| E | C3F2 (`c3f2`): compact ConvNeXt-style U-decoder, hybrid 2D/3D | 2.15 M |

- **Implementation:** details and deviations from MedCAGD in `README.md`, section "Step 2".
- **Calibration** (A100): D 0.52 s/it and 32.8 GiB peak, E 0.32 s/it and 22.0 GiB. Both are faster than A, so
  they train for the same 31,800 iterations. Actual training times: D 4.9 h, E 3.2 h, C seed 1 3.3 h.
- **Runs:** D seed 0, E seed 0 and a second seed of C. Full tables with C seed 0 as reference:
  `<experiment root>/evaluation/report_step2/report.md`.

### Results

| Run | Headline (tuned) | 2D | 3D | Headline (defaults) | Decode Z=8 | Peak memory Z=8 |
|---|---|---|---|---|---|---|
| A, seeds 0 / 1 | 0.292 / 0.297 | 0.365 | 0.219 / 0.230 | 0.280 / 0.302 | 232 ms | 8.9 GiB |
| C, seed 0 | 0.339 | **0.384** | 0.294 | **0.328** | **79 ms** | 4.5 GiB |
| C, seed 1 | **0.346** | **0.384** | 0.308 | 0.326 | **79 ms** | 4.5 GiB |
| D, seed 0 | 0.337 | 0.374 | 0.300 | 0.320 | 125 ms | 5.0 GiB |
| E, seed 0 | 0.342 | 0.371 | **0.312** | 0.323 | 93 ms | **4.1 GiB** |

- **Seed noise of C:** the two seeds differ by 0.007 on the headline, 0.0004 in 2D and 0.014 in 3D. Single
  datasets differ by up to 13 % (GoNuclear; Usiigaci +7 %, PlantSeg root +8 %), mostly by less than 8 %.
- **C** has the best headline and 2D results, the fastest decoder, and is the best or equal at default
  parameters.
- **E** has a headline between the two C seeds. It is 3.3 % lower in 2D, consistently across the 2D datasets.
  It is the best in 3D: +6 % against C seed 0, but only +1 % against C seed 1, i.e. within C's seed noise. The
  3D gain comes mostly from GoNuclear (+25 %) and NIS3D (+12 %); PlantSeg root is 11 % lower.
- **D** has a headline 0.002 below C seed 0 (within the seed spread), is 2.5 % lower in 2D and level in 3D. It has
  the largest per-dataset swings against C seed 0:

  | Dataset | D vs. C seed 0 |
  |---|---|
  | PlantSeg root | +25 % |
  | GoNuclear | +18 % |
  | EmbedSeg | +7 % |
  | CovidIF | +4 % |
  | CShaper | −15 % |
  | NIS3D | −14 % |
  | PlantSeg ovules | −8 % |
  | Usiigaci | −8 % |

  The CShaper and ovules losses come with many more merged objects (CShaper 54 % merged vs. 28 % for C, ovules
  67 % vs. 56 %). D is the largest and slowest of the new decoders.
- **Summary:** neither D nor E beats C overall. C and E are the candidates for training on all data: C is better
  in 2D, slightly faster and has two consistent seeds; E is somewhat better in 3D, but within C's seed noise.

### Caveats

- D and E have one seed each; their 3D differences to C are about the size of C's own seed spread.
- The tuned 3D parameters still transfer unevenly to the test sets. E's tuned 3D foreground weight (0.05) is at
  the edge of the search grid.
- The largest per-dataset swings come from small test sets: GoNuclear is a single volume; NIS3D and three
  EmbedSeg collections have 2 test crops each.

## Next steps and open items

- Train C or E on all training data. The decoders are to be implemented in torch-em first
  (`torch-em/HYBRID_UNETR_PLAN.md`) and then used from micro-sam.
- Tiled vs. untiled check for D and E (plan section 6), not implemented yet.
- Revisit the 3D post-processing tuning.
- Tuning saves results only after each complete round; with the 2 h QoS, B0's and C0's tuning timed out once.
  The tune stage now requests 4 h; saving after each chunk would make it robust.
