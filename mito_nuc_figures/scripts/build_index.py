"""Build the overview (README.md table and index.html gallery) from all per-dataset info.json files."""
import os
import html
import json
from glob import glob

from figure_utils import DATASET_GROUPS, FIGURE_ROOT


def load_infos():
    infos = []
    for path in sorted(glob(os.path.join(FIGURE_ROOT, "*", "*", "info.json"))):
        with open(path) as f:
            info = json.load(f)
        info["folder"] = os.path.relpath(os.path.dirname(path), FIGURE_ROOT)
        infos.append(info)
    return infos


def get_group(info):
    return info["folder"].split("/")[0]


def get_figures(folder):
    if os.path.basename(folder) == "cem_mitolab":
        figures = glob(os.path.join(FIGURE_ROOT, os.path.dirname(folder), "cem", "mitolab_*.png"))
    else:
        figures = glob(os.path.join(FIGURE_ROOT, folder, "**", "*.png"), recursive=True)
    if os.path.basename(folder) == "cem":
        figures = [path for path in figures if not os.path.basename(path).startswith("mitolab_")]
    return sorted(os.path.relpath(path, FIGURE_ROOT) for path in figures)


def as_text(value):
    if isinstance(value, (list, tuple)):
        return "; ".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {as_text(v)}" for k, v in value.items())
    return "" if value is None else str(value)


def table_cell(value):
    return as_text(value).replace("|", "/").replace("\n", " ")


def write_readme(infos):
    lines = [
        "# EM datasets with mitochondria / nucleus labels",
        "",
        "Datasets are split by label type into `mito_and_nucleus/`, `mito_only/` and `nucleus_only/`."
        " One folder per torch-em dataset with one figure per volume (rows: raw, then one overlay row per class;"
        " columns: selected z-slices plus a ZX view). Open `index.html` for the gallery with notes."
        " Scripts are in `scripts/`; rebuild this overview with `python scripts/build_index.py`.",
        "",
        "Read `FINDINGS.md` first: label problems, loader bugs and duplicates that matter for choosing training data.",
        "",
    ]
    for group in DATASET_GROUPS:
        members = [info for info in infos if get_group(info) == group]
        if not members:
            continue
        lines += [f"## {group}/ ({len(members)})", ""]
        lines += ["| Dataset | Modality | Dim | Labels | Volumes | Figures | Issues |", "|---|---|---|---|---|---|---|"]
        for info in members:
            row = [
                f"[{info['dataset']}]({info['folder']}/)", info.get("modality"), info.get("dimensionality"),
                info.get("label_quality"), info.get("n_volumes"), len(get_figures(info["folder"])), info.get("issues"),
            ]
            lines.append("| " + " | ".join(table_cell(v) for v in row) + " |")
        lines.append("")
    with open(os.path.join(FIGURE_ROOT, "README.md"), "w") as f:
        f.write("\n".join(lines))


def dataset_section(info):
    fields = ["modality", "dimensionality", "label_quality", "n_volumes", "data_path", "download", "notes"]
    rows = "".join(
        f"<tr><th>{key}</th><td>{html.escape(as_text(info.get(key)))}</td></tr>" for key in fields if info.get(key)
    )
    callouts = ""
    for key in ("observations", "issues"):
        if info.get(key):
            callouts += f'<p class="{key}"><b>{key}:</b> {html.escape(as_text(info[key]))}</p>'
    figures = "".join(
        f'<figure><a href="{html.escape(path)}" target="_blank"><img loading="lazy" src="{html.escape(path)}"></a>'
        f"<figcaption>{html.escape(path)}</figcaption></figure>"
        for path in get_figures(info["folder"])
    )
    return (
        f'<section id="{info["folder"]}"><h2>{html.escape(info["dataset"])}</h2>'
        f"<table>{rows}</table>{callouts}<details><summary>figures</summary><div class=\"grid\">{figures}</div>"
        "</details></section>"
    )


def write_html(infos):
    style = (
        ":root{--bg:#fff;--fg:#111;--muted:#666;--line:#ddd;--obs:#eef6ee;--iss:#fbeeee}"
        "@media (prefers-color-scheme: dark){:root{--bg:#161616;--fg:#eee;--muted:#999;--line:#333;"
        "--obs:#1d2a1d;--iss:#2e1d1d}}"
        "body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:0 16px 40px;max-width:1600px}"
        "nav{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}"
        "nav a{margin-right:10px;color:inherit}nav b{margin-right:6px}"
        "table{border-collapse:collapse;margin:6px 0}th,td{text-align:left;padding:2px 10px 2px 0;vertical-align:top}"
        "th{color:var(--muted);font-weight:normal}.observations{background:var(--obs);padding:6px}"
        ".issues{background:var(--iss);padding:6px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}"
        "img{width:100%;border:1px solid var(--line)}figure{margin:0}figcaption{color:var(--muted);font-size:12px}"
        "summary{cursor:pointer;margin:6px 0}section{border-top:1px solid var(--line);padding-top:8px}"
    )
    nav, body = "", ""
    findings = '<div><a href="FINDINGS.md">FINDINGS.md</a> (read first)</div>'
    for group in DATASET_GROUPS:
        members = [info for info in infos if get_group(info) == group]
        if not members:
            continue
        nav += f"<div><b>{group}/:</b>" + "".join(f'<a href="#{i["folder"]}">{i["dataset"]}</a>' for i in members)
        nav += "</div>"
        body += f"<h1>{group}/</h1>" + "".join(dataset_section(info) for info in members)
    page = (
        f'<!doctype html><html><head><meta charset="utf-8"><title>Mito and nucleus EM</title><style>{style}</style>'
        f"</head><body><nav>{nav}{findings}</nav>{body}</body></html>"
    )
    with open(os.path.join(FIGURE_ROOT, "index.html"), "w") as f:
        f.write(page)


def main():
    infos = load_infos()
    write_readme(infos)
    write_html(infos)
    print(f"Indexed {len(infos)} datasets.")


if __name__ == "__main__":
    main()
