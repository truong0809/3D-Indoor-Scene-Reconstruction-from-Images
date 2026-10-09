#!/usr/bin/env bash
# run_baseline_deepblending.sh - Buoc 4: tai lap 3DGS goc (Inria) tren Deep Blending.
# Tham so huan luyen / render / metrics giong full_eval.py cua Inria (doc tu file config).
# Chay lai an toan: canh nao da co results.json se duoc bo qua (FORCE=1 de chay lai).
#
# Cach chay (tren pod, sau khi setup_3dgs_inria.sh da xong):
#   bash scripts/run_baseline_deepblending.sh
#
# Bien tuy chon:
#   WS=/workspace  ENV_NAME=gs-inria  OUT_ROOT=<thu muc output>  FORCE=1
#   CONFIG=configs/baseline_deepblending.json

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_DIR/configs/baseline_deepblending.json}"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-inria}"
CODE_DIR="$WS/code/gaussian-splatting"
DATA_DIR="$WS/data"
OUT_ROOT="${OUT_ROOT:-$WS/outputs/baseline_inria/db}"
LOG_DIR="$WS/logs"
REPORT_DIR="$WS/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$OUT_ROOT" "$DATA_DIR/downloads"
LOG="$LOG_DIR/baseline_db_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
die()  { echo "ERROR: $*"; exit 1; }

MON_PID=""
cleanup() { if [ -n "$MON_PID" ]; then kill "$MON_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
step "0. Environment"
[ -r "$CONFIG" ] || die "missing config $CONFIG"
[ -d "$CODE_DIR" ] || die "missing $CODE_DIR - run scripts/setup_3dgs_inria.sh first"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (no GPU visible)"
# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
python -c "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; assert torch.cuda.is_available()" \
  || die "conda env $ENV_NAME is not ready - rerun scripts/setup_3dgs_inria.sh"

read_cfg() {  # read_cfg <dotted.key> -> value (lists are joined with spaces)
  python - "$CONFIG" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(" ".join(map(str, value)) if isinstance(value, list) else value)
PY
}
DATA_URL="$(read_cfg data.url)"
ARCHIVE_DIR="$DATA_DIR/$(read_cfg data.archive_dir)"
SUBDIR="$(read_cfg data.subdir)"
read -r -a SCENES <<< "$(read_cfg scenes)"
read -r -a TRAIN_ARGS <<< "$(read_cfg train_args)"
read -r -a RENDER_ITERS <<< "$(read_cfg render_iterations)"
read -r -a RENDER_ARGS <<< "$(read_cfg render_args)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
echo "config: $CONFIG"
echo "scenes: ${SCENES[*]} | GPU: ${GPU_NAME:-?} | output: $OUT_ROOT"

# ---------------------------------------------------------------------------
step "1. Dataset"
ZIP="$DATA_DIR/downloads/$(basename "$DATA_URL")"
locate_db_root() {  # the archive may or may not have a top-level folder
  if [ -d "$ARCHIVE_DIR/$SUBDIR" ]; then
    echo "$ARCHIVE_DIR/$SUBDIR"
  else
    find "$ARCHIVE_DIR" -maxdepth 3 -type d -name "$SUBDIR" 2>/dev/null | head -n 1 || true
  fi
}
DB_ROOT="$(locate_db_root)"
if [ -z "$DB_ROOT" ] || [ ! -d "$DB_ROOT/${SCENES[0]}" ]; then
  if [ ! -s "$ZIP" ]; then
    curl -fL --retry 3 -o "$ZIP.part" "$DATA_URL"
    mv "$ZIP.part" "$ZIP"
  fi
  sha256sum "$ZIP" | tee "$REPORT_DIR/$(basename "$ZIP").sha256"
  mkdir -p "$ARCHIVE_DIR"
  python -m zipfile -e "$ZIP" "$ARCHIVE_DIR"   # stdlib: no dependency on the unzip binary
  DB_ROOT="$(locate_db_root)"
fi
[ -n "$DB_ROOT" ] || die "cannot find the '$SUBDIR' folder inside $ARCHIVE_DIR"
echo "dataset root: $DB_ROOT"
for scene in "${SCENES[@]}"; do
  src="$DB_ROOT/$scene"
  [ -d "$src/sparse/0" ] && [ -d "$src/images" ] \
    || die "unexpected dataset layout: $src/sparse/0 or $src/images not found"
  echo "$scene: $(find "$src/images" -maxdepth 1 -type f | wc -l) images"
done

# ---------------------------------------------------------------------------
REPO_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
REPO_DIRTY="$(git -C "$REPO_DIR" status --porcelain 2>/dev/null | head -c 1 | wc -c)"
INRIA_COMMIT_NOW="$(git -C "$CODE_DIR" rev-parse HEAD)"

for scene in "${SCENES[@]}"; do
  src="$DB_ROOT/$scene"
  model="$OUT_ROOT/$scene"
  if [ -f "$model/results.json" ] && [ "${FORCE:-0}" != "1" ]; then
    step "$scene: results.json exists, skipping (FORCE=1 to rerun)"
    continue
  fi
  mkdir -p "$model"

  step "2. Train $scene"
  nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 2 > "$model/gpu_log.csv" 2>/dev/null &
  MON_PID=$!
  start_iso="$(date -Is)"
  t0="$(date +%s)"
  (cd "$CODE_DIR" && python train.py -s "$src" -m "$model" "${TRAIN_ARGS[@]}")
  t1="$(date +%s)"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  python - "$model/run_meta.json" "$scene" "$CONFIG" "${TRAIN_ARGS[*]}" "$start_iso" \
    "$((t1 - t0))" "${GPU_NAME:-unknown}" "$INRIA_COMMIT_NOW" "$REPO_COMMIT" "$REPO_DIRTY" <<'PY'
import json
import sys

import torch

out, scene, config, train_args, start, seconds, gpu, inria, repo, dirty = sys.argv[1:11]
meta = {
    "scene": scene,
    "config": config,
    "train_args": train_args,
    "train_start": start,
    "train_seconds": int(seconds),
    "gpu": gpu,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "inria_commit": inria,
    "repo_commit": repo,
    "repo_dirty": dirty.strip() != "0",
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=2)
print("train time: %.1f min" % (meta["train_seconds"] / 60))
PY

  step "3. Render $scene (iterations ${RENDER_ITERS[*]})"
  for it in "${RENDER_ITERS[@]}"; do
    (cd "$CODE_DIR" && python render.py --iteration "$it" -s "$src" -m "$model" "${RENDER_ARGS[@]}")
  done

  step "4. Metrics $scene"
  (cd "$CODE_DIR" && python metrics.py -m "$model")
  [ -f "$model/results.json" ] || die "metrics.py did not write $model/results.json"
done

# ---------------------------------------------------------------------------
step "5. Compare with Inria reference numbers"
SUMMARY="$REPORT_DIR/baseline_db_summary_$STAMP.json"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m indoor3d.eval.compare_reference --config "$CONFIG" --model-root "$OUT_ROOT" --out "$SUMMARY"

echo
echo "DONE. Log: $LOG"
echo "Send back: $SUMMARY (and $LOG if something failed)"
