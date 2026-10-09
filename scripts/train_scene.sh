#!/usr/bin/env bash
# train_scene.sh - Huan luyen mot canh bang gsplat theo mot che do trong configs/train_modes.json.
#
# Cach chay (tren pod, sau setup_gsplat.sh va prepare_scene.sh):
#   bash scripts/train_scene.sh <ten_canh | thu_muc_canh> <che_do> [tham so them cho indoor3d.train.run]
# Vi du:
#   bash scripts/train_scene.sh phong01 default      # baseline
#   bash scripts/train_scene.sh phong01 mcmc         # tu lay so Gaussian cuoi cua lan chay default cung canh
#   bash scripts/train_scene.sh phong01 mcmc_demo    # dung tham so cua notebook demo (khong vao so sanh cong bang)
#   TAG=thu bash scripts/train_scene.sh phong01 default --protocol max_steps=3000   # chay thu nhanh
#   TAG=lo001 BASELINE_TAG= bash scripts/train_scene.sh phong01 mcmc --set opacity_reg=0.001
#
# Bien tuy chon:
#   WS=/workspace  ENV_NAME=gs-gsplat  OUT_ROOT=$WS/outputs/modes  SEED=42
#   TAG=<ten>        bat buoc khi co --protocol / --set, de khong ghi vao thu muc cua luot chuan
#   BASELINE_TAG=    TAG cua luot default dung lam ngan sach cho mcmc (mac dinh = TAG; de trong = luot chuan)
# Ket qua: $OUT_ROOT/<canh>/<che_do>[_<TAG>]_s<seed>/ (summary.json, run_meta.json, train.log, eval.log, ...)

set -euo pipefail

if [ "$#" -lt 2 ]; then
  sed -n '2,20p' "$0"
  exit 1
fi
SCENE_ARG="$1"
MODE="$2"
shift 2

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
OUT_ROOT="${OUT_ROOT:-$WS/outputs/modes}"
SEED="${SEED:-42}"
TAG="${TAG:-}"
BASELINE_TAG="${BASELINE_TAG-$TAG}"
LOG_DIR="$WS/logs"

die() { echo "ERROR: $*"; exit 1; }

if [ -d "$SCENE_ARG" ]; then
  SCENE_DIR="$(cd "$SCENE_ARG" && pwd)"
else
  SCENE_DIR="$WS/data/scenes/$SCENE_ARG"
fi
[ -d "$SCENE_DIR" ] || die "scene not found: $SCENE_DIR"
NAME="$(basename "$SCENE_DIR")"
[[ "$MODE" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid mode name: $MODE"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer"
[[ "$TAG$BASELINE_TAG" =~ ^[A-Za-z0-9_-]*$ ]] || die "TAG may only contain letters, digits, '_' and '-'"
if [ -z "$TAG" ] && [[ " $* " == *" --protocol"* || " $* " == *" --set"* ]]; then
  die "--protocol / --set change the configuration: set TAG=<name> so the standard run is not overwritten"
fi
OUT="$OUT_ROOT/$NAME/${MODE}${TAG:+_$TAG}_s$SEED"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_${NAME}_${MODE}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
# chay thu fused-ssim tren GPU: ban 328dc98 chi duoc build cho kien truc GPU luc cai dat
python -c "import torch, fused_ssim; from gsplat import csrc; x = torch.rand(1, 3, 16, 16, device='cuda'); fused_ssim.fused_ssim(x, x)" 2>/dev/null \
  || die "conda env $ENV_NAME is not ready for this GPU - run scripts/setup_gsplat.sh (it rebuilds when the GPU type changes)"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=("$@")
needs_budget="$(python - "$REPO_DIR/configs/train_modes.json" "$MODE" <<'PY'
import json
import sys
mode = json.load(open(sys.argv[1], encoding="utf-8"))["modes"].get(sys.argv[2], {})
print("yes" if (mode.get("cap_max") or {}).get("policy") == "match_baseline" else "no")
PY
)"
if [ "$needs_budget" = "yes" ] && [[ " ${EXTRA[*]-} " != *" --cap-max"* ]]; then
  BASELINE="$OUT_ROOT/$NAME/default${BASELINE_TAG:+_$BASELINE_TAG}_s$SEED"
  [ -f "$BASELINE/summary.json" ] || die "mode $MODE needs a finished default run first: $BASELINE"
  EXTRA+=(--cap-max-from "$BASELINE")
fi

echo "scene: $SCENE_DIR | mode: $MODE | seed: $SEED | out: $OUT"
if [[ " ${EXTRA[*]-} " != *" --test-list"* ]]; then
  echo "NOTE: test images are every 8th frame of the same video (interpolation between near-identical views);"
  echo "      for a held-out trajectory pass --test-list <file> (docs/design/training_modes.md, 4.2)"
fi
python -m indoor3d.train.run --mode "$MODE" --scene "$SCENE_DIR" --scene-name "$NAME" --out "$OUT" \
  --seed "$SEED" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete "${EXTRA[@]}"
echo "log: $LOG"
