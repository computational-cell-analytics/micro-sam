# AIS optimization for the joint/v4 geodesic `hvit_t` model: summary

Concise findings of the 2026-09-06/07 campaign. The full decision log with every number, job id and run
directory is `AIS_V4_OPTIMIZATION.md`; the set-up is `EXPERIMENTAL_SETUP.md` (§13-14); the decoder-side
follow-up is `AIS_DECODER_TRAINING_PROPOSAL.md` at the repository root (uncommitted).

## Outcome

New `hvit_t` defaults for the flow (sparse) post-processing in `micro_sam/v2/postprocessing.py`:

| | old (registry) | new, images | new, volumes |
|---|---|---|---|
| density smoothing `sigma` | 0.5 | **1.0** | 0.5 |
| size floor `min_size` | 100 | **50** | 100 |
| instance filter `boundary_magnitude_max` | off | **0.4** | **0.4** |
| seed floor `seed_floor` | (monotone flooding) | 'none' (floors screened, not adopted) | 'none' |
| foreground 0.5, density 10, travel 25 px, foreground weight 0.5 | unchanged | | |

`drop_instances_without_boundary_dip` is new library logic: the geodesic decoder's distance magnitude falls
to zero along every real object boundary, so an instance whose boundary median exceeds the threshold is a
false foreground region (3 ms per image). `default_postprocessing` became dimension-aware (`sparse_volume`
overrides). The dense multicut is untouched.

## Results (mSA, new defaults against the old ones)

| instrument | old | new | change | datasets up |
|---|---:|---:|---:|---|
| 2D development corpus, 11 datasets | 0.3357 | 0.3437 | +2.4 % | 9 / 11 (worst −0.8 %) |
| 2D holdout, 5 datasets | 0.2337 | 0.2437 | +4.3 % | 5 / 5 |
| 3D LM deep crops, primary / holdout | 0.1765 / 0.1998 | 0.1847 / 0.2193 | +4.7 % / +9.7 % | 5 / 6, none down |
| 3D test manifest (7 test-only datasets, opened once) | 0.1083 | 0.1120 | +3.4 % | 6 / 6 scorable |
| production 2D test splits, 23 datasets | 0.2735 | 0.2864 | +4.7 % | 21 / 23 |
| of which the 12 never used for tuning | 0.2104 | 0.2191 | +4.2 % | 10 / 12 |
| production 3D LM test splits, 10 datasets | 0.1455 | 0.1507 | +3.6 % | 9 / 10, none down |

Known costs: microbeseg −11.4 % (0.142 → 0.126, caused by the wider smoothing alone, which merges its
small dense bacteria) and arvidsson −0.8 %. The comparator's quality gate (+5 % macro on the five primary
datasets) reads +3.9 % / +4.3 % with every dataset up and runtime within +3 %; the campaign's generalization
gate passes on every instrument. APG remains ahead (2D primary 0.296 vs 0.246; 3D crops 0.33 vs 0.18).

## What the diagnostics established

- The v4 field is centre-directed and its magnitude dips at object centres; the old post-processing was
  tuned to the v2 medial-axis field. Travel to convergence is *not* the fix (+7 % balanced but 5 / 16 up,
  more background seeds).
- Loss decomposition of the old defaults: merges dominate the touching-cell data (livecell 25 % merged +
  11 % absorbed, tissuenet 20 % + 5 %), background seeds dominate deepseas (2× the object count), deepbacs and
  neurips_cellseg; `min_size` 100 alone cost tissuenet 7.5 % of its matches.
- Oracles: a ground-truth ridge with the predicted seeds doubles livecell / deepbacs; ground-truth seeds add
  +5-20 % (+50-80 % on small-object data); the ground-truth foreground ceiling is the largest but mixes
  extent with separation.
- Mechanism of the merges: the watershed floods monotonically and every proper seed sits on a height peak
  (the centre dip), so a seed with a deeper dip than the contact loses its object. The same property
  suppresses spurious seeds, which is why a plain seed floor lost on the old defaults.
- What did not generalize (all recorded with numbers): relative / particle-count seeds (split large cells),
  trajectory assignment (inherits the blurred field), direction and divergence ridges (the network smooths
  the flip over 6-8 px), height-map transforms, a decoder-consistency merge (same-object vs different-object
  seed pairs are not separable), stronger volume settings (won on the tuning crops, lost on the test manifest).
- What did: wider density smoothing (merges the jittering sinks of large cells, removes one-pixel seeds),
  the ground-truth-like size floor once the spurious seeds are gone, and the boundary filter.

## Follow-up screen: seed floors on the promoted defaults (not adopted)

The one untested lever left by the diagnostics was to lower the height map under the seeds (so that the
monotone flooding cannot hold a seed's front at its own height), now that the promoted defaults remove the
spurious seeds that sank the same idea on the old defaults. Opt-in keyword `seed_floor` ('zero', 'ring').
Against the promoted defaults: 2D development −0.6 % (zero) / −1.3 % (ring), 4 and 3 of 11 datasets up
(livecell +2.5 %, tissuenet +5.5 %, deepbacs −6.7 %); 2D holdout −1.2 %; 3D LM crops −40 to −52 % (zero) and
−26 % (ring) with no source up: the floors halve the merges but release more instances than they recover,
and in 3D every large nucleus carries several weak sinks that all flood. **The A3 defaults stay for images
and volumes**; the keyword remains available. The remaining merges need a sharper field from the decoder
(`AIS_DECODER_TRAINING_PROPOSAL.md`).

## Where things are

- Library: `micro_sam/v2/postprocessing.py` (defaults, `drop_instances_without_boundary_dip`,
  `lower_height_under_seeds`, dimension-aware `default_postprocessing`); tests in
  `test/test_v2_automatic_segmentation.py`.
- Harness: `finetuning/v2/evaluation/optimization/benchmark_ais_optimization.py` (predict / run / screen /
  sweep / oracle / report on cached predictions), `ais_campaign_tasks.py`, `report_ais_sweep.py`,
  `report_ais_production.py`, configurations `configs/ais_*.json`; unit tests `test/test_ais_optimization.py`.
- Data: caches, run directories, sweeps, oracles and reports under `<root>/ais/`; production results under
  `experiments/v4_geodesic_ais_optimization/results/` (tags `old-defaults`, `a2-defaults` for 2D,
  `a3-defaults` for 3D).
- Open: the dense multicut (Phase 5.3: elf's `beta` cuts more when higher, the docstring says the opposite,
  and the EM crops shatter into 15k fragments), and the decoder-side changes of the proposal.
