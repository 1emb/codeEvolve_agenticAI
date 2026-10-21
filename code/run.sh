#!/bin/bash
# Launch CodeEvolve on one graph coloring problem.
#   bash run.sh <instance> <qwen|mock> [tag] [load_ckpt]
# Output goes to ~/codeEvolve/experiments/graph_coloring/<instance>/<tag>/ (symlink to sheshnagnobackup).
set -e
INSTANCE=${1:?instance}; MODE=${2:?qwen|mock}; TAG=${3:-$MODE}; LOAD_CKPT=${4:-0}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)            # codeEv repo root
PROB="$ROOT/problems/graph_coloring/$INSTANCE"
OUT="$HOME/codeEvolve/experiments/graph_coloring/$INSTANCE/$TAG"
mkdir -p "$OUT"
if [ "$MODE" = mock ]; then
  export API_BASE=test API_KEY=test
else
  export API_BASE=${API_BASE:-http://localhost:8000/v1} API_KEY=${API_KEY:-EMPTY}
fi
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate codeevolve
cd "$ROOT"
echo "codeevolve --inpt_dir=$PROB/input --cfg_path=$PROB/config_$MODE.yaml --out_dir=$OUT --load_ckpt=$LOAD_CKPT"
exec codeevolve --inpt_dir="$PROB/input" --cfg_path="$PROB/config_$MODE.yaml" --out_dir="$OUT" --load_ckpt="$LOAD_CKPT" --y
