#!/usr/bin/env bash
#
# run_pipeline.sh
#
# Thin wrapper around run_pipeline.py. All paths are passed straight through
# as command-line arguments -- nothing is hardcoded here. This exists mainly
# so you have one command to run (and one place to point a Katana job
# script at), plus a couple of basic sanity checks before spending time
# loading an 8B-parameter model only to fail on a typo'd path.
#
# Usage: identical to run_pipeline.py -- run with -h to see all options:
#   ./run_pipeline.sh -h
#
# Example (single record):
#   ./run_pipeline.sh \
#       --zeta-repo /path/to/Zeta \
#       --anyecg-repo /path/to/anyECG-chat-main \
#       --projection-ckpt /path/to/stage3_ckpt/projection.pth \
#       --ecg-model-ckpt /path/to/stage3_ckpt/ecg_model.pth \
#       --lora-ckpt /path/to/stage3_ckpt \
#       --record /path/to/some/record \
#       --output results.json
#
# Example (batch, one record path per line in records.txt):
#   ./run_pipeline.sh \
#       --zeta-repo /path/to/Zeta \
#       --anyecg-repo /path/to/anyECG-chat-main \
#       --projection-ckpt /path/to/stage3_ckpt/projection.pth \
#       --ecg-model-ckpt /path/to/stage3_ckpt/ecg_model.pth \
#       --lora-ckpt /path/to/stage3_ckpt \
#       --records-file records.txt \
#       --output results.json
#
# To adapt this into a Katana PBS job script: wrap the final `python3` call
# below with #PBS directives (queue, walltime, ngpus/mem select statement)
# at the top of this file, and submit with `qsub run_pipeline.sh`. Not
# added here since PBS resource requirements depend on your batch size and
# Katana project allocation -- ask if you want a PBS-specific version built.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Optional: activate a virtual environment if one is set. ---
# Uncomment and point at your venv, or export VENV_PATH before calling this
# script -- left off by default so this doesn't silently activate the wrong
# environment on Katana if you're using a module-loaded Python instead.
if [[ -n "${VENV_PATH:-}" ]]; then
    echo "Activating virtual environment: ${VENV_PATH}"
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" &> /dev/null; then
    echo "ERROR: '${PYTHON_BIN}' not found on PATH. Set PYTHON_BIN or activate an environment first." >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "ERROR: no arguments given. Run '$0 -h' for usage." >&2
    exit 1
fi

echo "Running pipeline with: ${PYTHON_BIN} ${SCRIPT_DIR}/run_pipeline.py $*"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py" "$@"
