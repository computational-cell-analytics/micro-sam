"""Create the overview table of the training data of the micro-sam2 EM organelle models (mitochondria and nuclei).

The table combines the two models: the mitochondria model ('em_mito', see ``EM_MITO_DATASETS`` in
``micro_sam/v2/datasets/generalist_loader.py`` and ``em_organelles/train_em_mito.py``) and the nucleus model
('em_nucleus', see ``EM_NUCLEUS_DATASETS`` and ``em_organelles/train_em_nucleus.py``). It has the columns of the
table of ``create_data_overview.py``:

- ``dataset``: The dataset name. It follows the dataset names of the loader, except for vEM nuclei, which has a 2d
  and a 3d row. Rows with test data only use the dataset name of Figure 6, except for 'mitolab_3d'.
- ``modality``: EM; the soft X-ray tomography of SXT INS-1E is X-ray.
- ``structure``: The labeled structure, mitochondrion or nucleus. It also tells which model the dataset trains.
- ``dimensionality``: 2d or 3d.
- ``role``: ``train`` (training data only), ``ID`` or ``OOD`` (scored test data), following the ``PANEL_DATASETS``
  of the Figure 6 plotting script in ``scripts/plotting/v2/for_paper``. An ID row has training data if the test data
  comes from the same source as a training dataset. The ID rows without training data (parlakgul_liver_eval,
  dauer_larva_nucleus) are other samples of a dataset whose other samples train the model.
- ``train_images_or_volumes``, ``train_objects``: The images or volumes and the labeled objects of the training
  data (train + val split), counted with ``micro_sam.v2.datasets.data_counter``. So they follow the splits and the
  label pre-processing of the loader. See its docstring for how stacks and regions of interest count.
- ``test_images_or_volumes``, ``test_objects``: The scored test crops and their objects, counted as for
  ``create_data_overview.py``. A dataset that Figure 6 pools from several evaluation datasets (``POOLED``, e.g. the
  MitoEM 2.0 volumes) sums the counts of its parts.
- ``comment``: Notes, e.g. on the subsets or the annotation of a dataset.

The table (``data-overview/data_overview_em_organelles.csv`` by default) is also the cache, as for
``create_data_overview.py``: the counts are only computed for rows whose count cells are empty, or for all requested
rows with ``--force``. Building the training datasets of a domain takes several minutes.

Usage:

    python create_data_overview_em_organelles.py                    # fill in the missing counts
    python create_data_overview_em_organelles.py --datasets lucchi  # only process this dataset
    python create_data_overview_em_organelles.py --datasets lucchi --force  # recompute its counts
"""

import os
import argparse
from typing import Callable

from create_data_overview import (
    DEFAULT_DATA_ROOT, REPO_ROOT, SCRIPT_DIR, Entry, Train, _path_has, check_registry, read_figure_literal,
    read_panels, run,
)

DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "data-overview", "data_overview_em_organelles.csv")
FIGURE_SCRIPTS = {3: os.path.join(REPO_ROOT, "scripts", "plotting", "v2", "for_paper", "plots_for_figure6.py")}

EM, XRAY = "EM", "X-ray"
MITO, NUCLEUS = "mitochondrion", "nucleus"


def _ndim_is(ndim: int) -> Callable:
    """Select the label sources of this dimensionality."""
    return lambda unit: unit.ndim == ndim


def _mito(folder, select=None):
    return Train("em_mito", folder, select)


def _nucleus(folder, select=None):
    return Train("em_nucleus", folder, select)


ENTRIES = [
    # Mitochondria, 2d.
    Entry("mitolab", EM, MITO, 2, _mito("mitolab"),
          comment="CEM-MitoLab: tiles from about 600 source volumes and images. The groups whose names match a "
          "mitochondria evaluation dataset are left out."),
    Entry("probtem", EM, MITO, 2, _mito("probtem_full")),
    Entry("u2os_fibsem_mito", EM, MITO, 2, _mito("u2os_fibsem"), comment="Single FIB-SEM slices of a U2-OS cell."),
    Entry("vnc", EM, MITO, 2, _mito("vnc"), comment="Single slices of the 20-slice ssTEM train stack."),
    # Mitochondria, 3d.
    Entry("aive_mito", EM, MITO, 3, _mito("aive")),
    Entry("asem_mito", EM, MITO, 3, _mito("asem"), test="asem_mito", comment="Boxes of the volumes 46, 58 and 2E."),
    Entry("brain_pfibsem_mito", EM, MITO, 3, _mito("brain_pfibsem"),
          comment="The noisy stack is registered to the reference stack and shares its labels."),
    Entry("cellmap_mito", EM, MITO, 3, _mito("cellmap_mito_crops"),
          comment="Crops of the CellMap Segmentation Challenge."),
    Entry("densecell_mito", EM, MITO, 3, _mito("densecell"), comment="The mitochondria class of the semantic labels."),
    Entry("hela_stard3_mito", EM, MITO, 3, _mito("hela_stard3")),
    Entry("human_organoids_mito", EM, MITO, 3, _mito("human_organoids_mito"),
          comment="The instances are precomputed with a neck watershed (em_organelles/preprocess_mito_dataset.py)."),
    Entry("lucchi", EM, MITO, 3, _mito("lucchi"), test="lucchi",
          comment="Lucchi++, the re-annotated Lucchi train volume."),
    Entry("microns_pinky_mito", EM, MITO, 3, _mito("microns"),
          comment="The densely annotated stitched volume vol19-vol34 of MICrONS pinky."),
    Entry("mito_segem", EM, MITO, 3, _mito("mito_segem"),
          comment="The testis and brainstem boxes. The intestine box is evaluation data."),
    Entry("mitoem", EM, MITO, 3, _mito("mitoem"), test="mitoem", comment="The train volumes of MitoEM-H and MitoEM-R."),
    Entry("mitoemv2", EM, MITO, 3, _mito("mitoemv2"), test="mitoemv2",
          comment="MitoEM 2.0: beta, jurkat, macro, podo, sperm and stem. pyra repeats MitoEM-H and is left out. "
          "The test data also has the mossy fiber boutons (mitoemv2_mossy), which Figure 6 pools into MitoEM 2.0."),
    Entry("mitoemv2_mossy", EM, MITO, 3, _mito("mitoverse_derived", _path_has("/han24_")),
          comment="The mossy fiber boutons of MitoEM 2.0 (MitoVerse han24). Only the boutons are annotated."),
    Entry("mitoverse_haberl18", EM, MITO, 3, _mito("mitoverse", _path_has("/haberl18/"))),
    Entry("mitoverse_microns1", EM, MITO, 3, _mito("mitoverse_derived", _path_has("/turner22_")),
          comment="MICrONS phase 1."),
    Entry("mitoverse_xiao18", EM, MITO, 3, _mito("mitoverse", _path_has("/xiao18/"))),
    Entry("parlakgul_fasting_liver_mito", EM, MITO, 3, _mito("parlakgul_fasting_liver"),
          comment="Other samples than the Parlakgul liver samples of the evaluation."),
    Entry("snemi_mito", EM, MITO, 3, _mito("snemi_mito"), test="snemi_mito"),
    Entry("uro_cell", EM, MITO, 3, _mito("uro_cell"), comment="The 5 of 8 UroCell volumes with mitochondria labels."),
    # Nuclei, 2d.
    Entry("cardiomyocyte_tem_nucleus", EM, NUCLEUS, 2, _nucleus("cardiomyocyte_tem_nucleus")),
    Entry("perez_nucleus", EM, NUCLEUS, 2, _nucleus("perez_nucleus"),
          comment="The test tiles of the dataset are the validation split of the training."),
    Entry("tumor_spheroid_nuclei", EM, NUCLEUS, 2, _nucleus("tumor_spheroid_nucleus"),
          comment="Single SBF-SEM slices."),
    Entry("vem_nuclei_2d", EM, NUCLEUS, 2, _nucleus("vem_nuclei_nucleus", _ndim_is(2)),
          comment="vEM nuclei, array tomography SEM. Proofread NucleoNet predictions."),
    Entry("waenet_sstem_nucleus", EM, NUCLEUS, 2, _nucleus("waenet_nucleus", _path_has("/dataset_3_")),
          comment="WAE-Net dataset 3: serial ssTEM sections of one cell."),
    Entry("waenet_tem_nucleus", EM, NUCLEUS, 2, _nucleus("waenet_nucleus", _path_has("/tem_")),
          comment="WAE-Net datasets 4-7: TEM images of lymphocytes and B cells. Only the central cell is labeled."),
    # Nuclei, 3d.
    Entry("betaseg_nucleus", EM, NUCLEUS, 3, _nucleus("betaseg_nucleus_boxes"),
          comment="Only the nucleus of one target cell per volume is labeled."),
    Entry("cellmap_nucleus", EM, NUCLEUS, 3, _nucleus("cellmap_nucleus_crops"), test="cellmap_nucleus",
          comment="Crops of the CellMap Segmentation Challenge. The test data is the held-out crop357."),
    Entry("cerebellum_nucleus", EM, NUCLEUS, 3, _nucleus("cerebellum_fibsem_nucleus")),
    Entry("fafb_nuclei", EM, NUCLEUS, 3, _nucleus("fafb_nuclei"), test="fafb_nuclei",
          comment="The labels are the automatic FlyWire segmentation."),
    Entry("human_organoids_nucleus", EM, NUCLEUS, 3, _nucleus("human_organoids_nucleus"),
          test="human_organoids_nucleus",
          comment="The test data is the validation part of the volumes (the last 10% of the slices), which is also in "
          "the train + val counts."),
    Entry("janelia_nucleus", EM, NUCLEUS, 3, _nucleus("janelia_nucleus"),
          comment="Cellpose pseudo-labels, except for the 3 manually labeled jrc_mus-nacc volumes."),
    Entry("macrophage_a431_nucleus", EM, NUCLEUS, 3, _nucleus("macrophage_a431_nucleus"),
          comment="Quick semantic annotations."),
    Entry("microns_nuclei", EM, NUCLEUS, 3, _nucleus("microns_nuclei"), test="microns_nuclei",
          comment="The labels are the automatic MICrONS minnie65 segmentation."),
    Entry("microsam_em_nuclei", EM, NUCLEUS, 3, _nucleus("microsam_em_nuclei"),
          comment="The train crop of the microSAM user study."),
    Entry("openorganelle_nucleus", EM, NUCLEUS, 3, _nucleus("openorganelle_nucleus"), comment="Automatic labels."),
    Entry("platynereis_nuclei", EM, NUCLEUS, 3, _nucleus("platynereis"), test="platynereis_nuclei"),
    Entry("sxt_nucleus", XRAY, NUCLEUS, 3, _nucleus("sxt_ins1e_mito"),
          comment="SXT INS-1E: soft X-ray tomography (EM-like), not EM. Semi-automatic labels."),
    Entry("vem_nuclei_3d", EM, NUCLEUS, 3, _nucleus("vem_nuclei_nucleus", _ndim_is(3)),
          comment="vEM nuclei, FIB-SEM. Proofread NucleoNet predictions."),
    Entry("waenet_fibsem_nucleus", EM, NUCLEUS, 3, _nucleus("waenet_nucleus", _path_has("/dataset_1.h5")),
          comment="WAE-Net dataset 1: FIB-SEM of pancreatic carcinoid cells."),
    # Mitochondria, test data only.
    Entry("parlakgul_liver_eval", EM, MITO, 3, test="parlakgul_liver_eval",
          comment="The manual Parlakgul liver evaluation crops. Other Parlakgul liver samples train the model."),
    Entry("cerebellum_fibsem_mito", EM, MITO, 3, test="cerebellum_fibsem_mito"),
    Entry("cmito", EM, MITO, 3, test="cmito", comment="The CMito volumes L1, L2, L3, adult and dauer, pooled."),
    Entry("human_liver_em", EM, MITO, 3, test="human_liver_em"),
    Entry("kasthuri", EM, MITO, 3, test="kasthuri"),
    Entry("macrophage_a431_mito", EM, MITO, 3, test="macrophage_a431_mito"),
    Entry("mitolab_3d", EM, MITO, 3, test="mitolab",
          comment="The CEM-MitoLab volumes c_elegans, fly_brain, glycolytic_muscle, hela_cell and salivary_gland, "
          "pooled. They are left out of the training data (see mitolab)."),
    # Nuclei, test data only.
    Entry("dauer_larva_nucleus", EM, NUCLEUS, 3, test="dauer_larva_nucleus",
          comment="Boxes of the OpenOrganelle jrc_dauer-larva volume, with automatic COSEM nucleus instances. Other "
          "OpenOrganelle volumes train the model."),
    Entry("aive_nucleus", EM, NUCLEUS, 3, test="aive_nucleus"),
    Entry("etchacell_nucleus", EM, NUCLEUS, 3, test="etchacell_nucleus"),
    Entry("human_liver_em_nucleus", EM, NUCLEUS, 3, test="human_liver_em_nucleus"),
    Entry("malecns_nuclei", EM, NUCLEUS, 3, test="malecns_nuclei"),
    Entry("placenta_nucleus", EM, NUCLEUS, 3, test="placenta_nucleus"),
    Entry("zheng_ca3_nuclei", EM, NUCLEUS, 3, test="zheng_ca3_nuclei"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Create the overview table of the micro-sam2 EM organelle training and evaluation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="The CSV table, also used as the cache.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="The root folder of the data.")
    parser.add_argument("-d", "--datasets", nargs="+", help="Only process these datasets. All by default.")
    parser.add_argument("-f", "--force", action="store_true", help="Recompute the counts that the table has.")
    parser.add_argument(
        "-n", "--n_workers", type=int, default=min(8, len(os.sched_getaffinity(0))),
        help="The number of processes that count in parallel. Use at most 8 on a 64 GB allocation.",
    )
    args = parser.parse_args()

    roles = check_registry(ENTRIES, read_panels(FIGURE_SCRIPTS))
    pooled = read_figure_literal(FIGURE_SCRIPTS[3], "POOLED")
    run(
        args.output, args.data_root, args.datasets, args.force, args.n_workers, registry=ENTRIES, roles=roles,
        pooled=pooled,
    )


if __name__ == "__main__":
    main()
