#!/bin/bash
# Run the runtime benchmark of Extended Data Figure 2 on an Apple silicon Mac (device 'mps').
#
# Before running, copy from the cluster:
#   1. the samples: <RUNTIME_BENCHMARK_ROOT on the cluster>/data  ->  $RUNTIME_BENCHMARK_ROOT/data
#      (/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/runtime_benchmark/data)
#   2. the micro-sam2 '_cells' models: ~/.cache/micro_sam/models/hvit_{t,s,b,l}_cells{,_decoder}
#      -> $MICRO_SAM2_MODEL_DIR
#   3. only to also time CREMI (not in the figure): the SegNeuron repository and its checkpoint, see SEGNEURON_ROOT and
#      SEGNEURON_CHECKPOINT, and pass '--only livecell gonuclear cremi'.
# The environment needs micro-sam (with the micro-sam2 dependencies), cellpose 4 and h5py.
#
# The results go to $RUNTIME_BENCHMARK_ROOT/results/m1. Copy that folder back to the cluster's 'results' folder and
# run 'python benchmark_runtimes.py collect' there. A run that is stopped resumes per sample. By default only the
# datasets of the figure are timed (LIVECell, GoNuclear); arguments after the script replace that choice.
set -e
export RUNTIME_BENCHMARK_ROOT=${RUNTIME_BENCHMARK_ROOT:-$HOME/runtime_benchmark}
export MICRO_SAM2_MODEL_DIR=${MICRO_SAM2_MODEL_DIR:-$HOME/.cache/micro_sam/models}
export SEGNEURON_ROOT=${SEGNEURON_ROOT:-$HOME/SegNeuron}
export SEGNEURON_CHECKPOINT=${SEGNEURON_CHECKPOINT:-$HOME/SegNeuron/SegNeuronModel.ckpt}
export PYTORCH_ENABLE_MPS_FALLBACK=1
DEVICE_TAG=${DEVICE_TAG:-m1}
cd "$(dirname "$0")"
if [ $# -eq 0 ]; then
    set -- --only livecell gonuclear
fi
python benchmark_runtimes.py run --all --device_tag "$DEVICE_TAG" --device mps "$@"
