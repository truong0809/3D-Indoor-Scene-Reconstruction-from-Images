#!/usr/bin/env bash
# prepare_scene.sh - Buoc 5: video (hoac thu muc anh) -> chon frame -> COLMAP -> bao cao chat luong.
#
# Cach chay (tren pod, sau setup_3dgs_inria.sh va setup_data_tools.sh):
#   bash scripts/prepare_scene.sh <video.mp4 | thu_muc_anh> <ten_canh> [tham so them cho frames.py]
# Vi du:
#   bash scripts/prepare_scene.sh /workspace/data/raw/bedroom01_20261012_main_1.mp4 bedroom01 --target 250
#
# Bien tuy chon:
#   WS=/workspace  ENV_NAME=gs-inria  TOOLS_ENV=gs-tools
#   COLMAP_ARGS="--matcher sequential --no-gpu ..."   (tham so them cho colmap_runner)
#   OVERWRITE=1                                       (xoa ket qua cu cua canh)

set -euo pipefail

if [ "$#" -lt 2 ]; then
  sed -n '2,13p' "$0"
  exit 1
fi
SRC="$1"
NAME="$2"
shift 2

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-inria}"
TOOLS_ENV="${TOOLS_ENV:-gs-tools}"
CONDA_DIR="$WS/miniforge3"
TOOLS_BIN="$CONDA_DIR/envs/$TOOLS_ENV/bin"
SCENE="$WS/data/scenes/$NAME"
LOG_DIR="$WS/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$SCENE"
LOG="$LOG_DIR/prepare_scene_${NAME}_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

die() { echo "ERROR: $*"; exit 1; }

[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "scene name may only contain letters, digits, '.', '_' and '-'"
[ -x "$TOOLS_BIN/colmap" ] && [ -x "$TOOLS_BIN/ffmpeg" ] || die "run scripts/setup_data_tools.sh first"
# shellcheck source=/dev/null
. "$CONDA_DIR/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OVERWRITE_FLAG=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "===== 1. Frames -> $SCENE/input ====="
if [ -f "$SRC" ]; then
  python -m indoor3d.data.frames --video "$SRC" --scene-dir "$SCENE" --ffmpeg "$TOOLS_BIN/ffmpeg" \
    "${OVERWRITE_FLAG[@]}" "$@"
elif [ -d "$SRC" ]; then
  python -m indoor3d.data.frames --images "$SRC" --scene-dir "$SCENE" "${OVERWRITE_FLAG[@]}" "$@"
else
  die "source not found: $SRC"
fi

echo
echo "===== 2. COLMAP ====="
read -r -a EXTRA_COLMAP_ARGS <<< "${COLMAP_ARGS:-}"
python -m indoor3d.sfm.colmap_runner --scene "$SCENE" --colmap "$TOOLS_BIN/colmap" \
  "${OVERWRITE_FLAG[@]}" "${EXTRA_COLMAP_ARGS[@]}"

echo
echo "DONE. Scene: $SCENE | reports: $SCENE/frames_report.json, $SCENE/sfm_report.json | log: $LOG"
echo "Train a first model to look at (all frames, no test split):"
echo "  cd $WS/code/gaussian-splatting && python train.py -s $SCENE -m $WS/outputs/scenes/$NAME --disable_viewer"
echo "Or train the two gsplat modes with a held-out test split (docs/design/training_modes.md):"
echo "  bash scripts/train_scene.sh $NAME default && bash scripts/train_scene.sh $NAME mcmc"
