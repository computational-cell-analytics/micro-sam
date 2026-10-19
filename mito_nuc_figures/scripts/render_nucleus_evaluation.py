"""Render the nucleus evaluation figures as the evaluation loads them (load_data of finetuning/v2/evaluation/common.py).

The evaluation scores the fixed evaluation crops of eval_crops_3d.json. Thus, these figures replace the earlier ones,
which read the nucleus volumes directly before the nucleus panel was registered.
The rendering is the one of render_mito_overview.py.
"""
import os
import argparse

import render_mito_overview as overview


IN_DOMAIN_EVALUATION = {
    "cellmap": ("cellmap (nuc)", ["cellmap_nucleus"]),
    "fafb_nuclei": ("fafb_nuclei", ["fafb_nuclei"]),
    "microns_nuclei": ("microns_nuclei", ["microns_nuclei"]),
    "platynereis": ("platynereis (nuclei)", ["platynereis_nuclei"]),
}
OOD_EVALUATION = {
    "human_liver_em": ("human_liver_em (nuc)", ["human_liver_em_nucleus"]),
    "u2os_fibsem": ("u2os_fibsem (nuc, new)", ["u2os_fibsem_nucleus"]),
    "etchacell": ("etchacell (nuc, new)", ["etchacell_nucleus"]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=None, help="Only render these figures.")
    args = parser.parse_args()
    overview.FIGURE_ROOT = os.path.join(overview.REPO, "mito_nuc_figures", "nuc")
    overview.RATIONALE_SECTION = "## Nucleus"
    overview.render_evaluation("in_domain_evaluation", IN_DOMAIN_EVALUATION, args.names)
    overview.render_evaluation("ood_evaluation", OOD_EVALUATION, args.names)


if __name__ == "__main__":
    main()
