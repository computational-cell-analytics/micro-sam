import argparse
from subprocess import run

parser = argparse.ArgumentParser()
parser.add_argument("--out", "-o", action="store_true")
args = parser.parse_args()

logo_url = "https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/micro-sam-logo.png"
cmd = ["pdoc", "--docformat", "google", "--logo", logo_url]

if args.out:
    cmd.extend(["--out", "tmp/"])
cmd.append("micro_sam")

run(cmd)

# pdoc --docformat google --logo "https://raw.githubusercontent.com/computational-cell-analytics/micro-sam/master/doc/images/micro-sam-logo.png" micro_sam
