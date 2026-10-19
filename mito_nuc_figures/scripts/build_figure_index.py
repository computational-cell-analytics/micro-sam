"""Write index.html and README.md of mito_nuc_figures for the figure layout <model>/<category>/<dataset>.png."""
import os

FIGURE_ROOT = "/mnt/vast-kisski/home/archit/u28048/micro-sam/mito_nuc_figures"
MODELS = {"mito": "Mitochondria model", "nuc": "Nucleus model"}
CATEGORIES = {
    "training": "Training",
    "in_domain_evaluation": "In-domain evaluation",
    "ood_evaluation": "Out-of-domain evaluation",
    "discarded": "Discarded",
}


def figures(model, category):
    folder = os.path.join(FIGURE_ROOT, model, category)
    return sorted(name for name in os.listdir(folder) if name.endswith(".png")) if os.path.isdir(folder) else []


def write_html():
    lines = [
        "<!doctype html>", "<html><head><meta charset='utf-8'><title>Mito and nucleus EM data</title>",
        "<style>body{font-family:sans-serif;margin:16px} img{max-width:100%;border:1px solid #ccc} "
        "details{margin:4px 0} summary{cursor:pointer;font-weight:bold}</style></head><body>",
        "<h1>Mito and nucleus EM data, as the loaders see it</h1>",
        "<p>Decisions and rationales: DECISIONS.md. Training figures are drawn from the em_mito / em_nucleus "
        "training loaders, evaluation figures from the evaluation loader (load_data of common.py).</p>",
    ]
    for model, model_title in MODELS.items():
        lines.append(f"<h2>{model_title}</h2>")
        for category, category_title in CATEGORIES.items():
            names = figures(model, category)
            lines.append(f"<h3>{category_title} ({len(names)})</h3>")
            for name in names:
                path = f"{model}/{category}/{name}"
                image = f"<a href='{path}'><img src='{path}'></a>"
                lines.append(f"<details><summary>{name[:-4]}</summary>{image}</details>")
    lines.append("</body></html>")
    with open(os.path.join(FIGURE_ROOT, "index.html"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_readme():
    lines = [
        "# Mito and nucleus EM data",
        "",
        "Figures of the data as the loaders see it, per model and decision: `<model>/<category>/<dataset>.png` with",
        "model `mito` or `nuc` and category `training`, `in_domain_evaluation`, `ood_evaluation` or `discarded`.",
        "Open `index.html` to browse them. Decisions and rationales are in `DECISIONS.md`, the survey in",
        "`FINDINGS.md`.",
        "",
    ]
    for model, model_title in MODELS.items():
        counts = ", ".join(f"{CATEGORIES[c].lower()} {len(figures(model, c))}" for c in CATEGORIES)
        lines.append(f"- {model_title} (`{model}/`): {counts}")
    with open(os.path.join(FIGURE_ROOT, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    write_html()
    write_readme()


if __name__ == "__main__":
    main()
