#!/usr/bin/env bash
# setup_data_tools.sh - Cai COLMAP va ffmpeg vao conda env rieng "gs-tools" trong /workspace.
# Tach rieng khoi env gs-inria de tranh xung dot thu vien.
# Chay lai an toan: env da co se duoc giu nguyen.
#
# Cach chay (tren pod, sau setup_3dgs_inria.sh vi can Miniforge da cai):
#   bash scripts/setup_data_tools.sh
# Bien tuy chon: WS=/workspace  TOOLS_ENV=gs-tools  COLMAP_BUILD=cpu (neu ban CUDA khong cai duoc)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${CONF:-$REPO_DIR/environment/gs-tools.conf}"
[ -r "$CONF" ] || { echo "ERROR: missing $CONF" >&2; exit 1; }
# shellcheck source=/dev/null
. "$CONF"
COLMAP_BUILD="${COLMAP_BUILD_OVERRIDE:-$COLMAP_BUILD}"

WS="${WS:-/workspace}"
TOOLS_ENV="${TOOLS_ENV:-gs-tools}"
CONDA_DIR="$WS/miniforge3"
LOG_DIR="$WS/logs"
REPORT_DIR="$WS/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
LOG="$LOG_DIR/setup_data_tools_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
die()  { echo "ERROR: $*"; exit 1; }

step "0. Preflight"
[ -x "$CONDA_DIR/bin/conda" ] || die "Miniforge not found in $CONDA_DIR - run scripts/setup_3dgs_inria.sh first"
case "$COLMAP_BUILD" in
  cuda|cpu) ;;
  *) die "COLMAP_BUILD must be 'cuda' or 'cpu' (got '$COLMAP_BUILD')" ;;
esac
# shellcheck source=/dev/null
. "$CONDA_DIR/etc/profile.d/conda.sh"

step "1. Conda env $TOOLS_ENV: colmap=$COLMAP_VERSION ($COLMAP_BUILD build) + ffmpeg"
if conda env list | awk '{print $1}' | grep -qx "$TOOLS_ENV"; then
  echo "env already exists"
else
  conda create -y -n "$TOOLS_ENV" -c conda-forge "colmap=$COLMAP_VERSION=${COLMAP_BUILD}*" ffmpeg \
    || die "conda could not install colmap ($COLMAP_BUILD build). If the driver is too old for the CUDA build, rerun with COLMAP_BUILD_OVERRIDE=cpu"
fi
ENV_BIN="$CONDA_DIR/envs/$TOOLS_ENV/bin"

step "2. Verify"
[ -x "$ENV_BIN/colmap" ] || die "colmap not found in $ENV_BIN"
[ -x "$ENV_BIN/ffmpeg" ] || die "ffmpeg not found in $ENV_BIN"
COLMAP_HEADER="$("$ENV_BIN/colmap" -h 2>&1 | grep -m 1 -i 'colmap' || true)"
echo "colmap: $COLMAP_HEADER"
echo "ffmpeg: $("$ENV_BIN/ffmpeg" -version | head -n 1)"
if [ "$COLMAP_BUILD" = "cuda" ] && ! printf '%s' "$COLMAP_HEADER" | grep -qi 'with CUDA'; then
  echo "WARN: COLMAP header does not mention CUDA; feature extraction may fall back to CPU"
fi
"$ENV_BIN/colmap" feature_extractor -h 2>&1 | grep -oE '(Feature|Sift)Extraction\.use_gpu' | head -n 1 \
  | sed 's/^/GPU option name: /'

step "3. Record environment"
conda env export -n "$TOOLS_ENV" --no-builds > "$REPORT_DIR/conda_env_${TOOLS_ENV}_$STAMP.yml"
{
  echo "date: $(date -Is)"
  echo "colmap: $COLMAP_HEADER"
  echo "ffmpeg: $("$ENV_BIN/ffmpeg" -version | head -n 1)"
  echo "conf: $CONF (COLMAP_VERSION=$COLMAP_VERSION, COLMAP_BUILD=$COLMAP_BUILD)"
} | tee "$REPORT_DIR/setup_data_tools_$STAMP.txt"

echo
echo "DONE. colmap: $ENV_BIN/colmap | ffmpeg: $ENV_BIN/ffmpeg | log: $LOG"
