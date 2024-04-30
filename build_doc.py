import argparse
import glob
import os
from subprocess import run


def check_docs_completeness():
    """@private
    All markdown and RST documentation files **MUST** be included in the module
    docstring at micro_sam/__init__.py
    """
    import micro_sam

    markdown_doc_files = glob.glob("doc/**/*.md", recursive=True)
    rst_doc_files = glob.glob("doc/**/*.rst", recursive=True)
    all_doc_files = markdown_doc_files + rst_doc_files
    missing_from_docs = [f for f in all_doc_files if os.path.basename(f) not in micro_sam.__doc__]
    if len(missing_from_docs) > 0:
        raise RuntimeError(
            "Documentation files missing! Please add include statements "
            "to the docstring in micro_sam/__init__.py for every file, eg:"
            "'.. include:: ../doc/filename.md'. "
            "List of missing files: "
            f"{missing_from_docs}"
        )


if __name__ == "__main__":
    check_docs_completeness()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "-o", action="store_true")
    args = parser.parse_args()

    logo_url = "https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/logo/logo_and_text.png"
    cmd = ["pdoc", "--docformat", "google", "--logo", logo_url]

    if args.out:
        cmd.extend(["--out", "tmp/"])
    cmd.append("micro_sam")

    run(cmd)

    # pdoc --docformat google --logo "https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/micro-sam-logo.png" micro_sam
