#!/usr/bin/env bash
# run_modes_deepblending.sh - Kiem chung hai che do huan luyen gsplat tren Deep Blending:
#   1. default (baseline B1) cho tung canh;
#   2. mcmc voi cung so Gaussian cuoi cua lan chay default (giao thuc cua paper 3DGS-MCMC);
#   3. bang so sanh default/mcmc; doi chieu default voi so lieu Inria cong bo (configs/baseline_deepblending.json)
#      va voi luot Inria tu chay cua run_baseline_deepblending.sh (tung canh), neu da co.
# Dung chung du lieu /workspace/data/tandt_db voi run_baseline_deepblending.sh (tai neu chua co).
# Chay lai an toan: lan chay nao da co summary.json OK se duoc bo qua.
#
# Cach chay (tren pod, sau setup_gsplat.sh):
#   bash scripts/run_modes_deepblending.sh
#
# Bien tuy chon:
#   WS=/workspace  ENV_NAME=gs-gsplat  OUT_ROOT=$WS/outputs/modes_db
#   SEEDS="42"              vd. SEEDS="42 43 44" de chay lap (docs/plan.md muc 4.2)
#   MCMC_DB_SET="opacity_reg=0.001"   paper 3DGS-MCMC dung lambda_o = 0.001 cho Deep Blending (de trong = 0.01)
#   CONFIG=configs/baseline_deepblending.json
#   B0_ROOT=$WS/outputs/baseline_inria/db   ket qua Inria tu chay de doi chieu tung canh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_DIR/configs/baseline_deepblending.json}"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
OUT_ROOT="${OUT_ROOT:-$WS/outputs/modes_db}"
B0_ROOT="${B0_ROOT:-$WS/outputs/baseline_inria/db}"
SEEDS="${SEEDS:-42}"
MCMC_DB_SET="${MCMC_DB_SET-opacity_reg=0.001}"
DATA_DIR="$WS/data"
LOG_DIR="$WS/logs"
REPORT_DIR="$WS/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$OUT_ROOT" "$DATA_DIR/downloads"
LOG="$LOG_DIR/modes_db_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
die()  { echo "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------------------
step "0. Environment"
[ -r "$CONFIG" ] || die "missing config $CONFIG"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (no GPU visible)"
# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
# chay thu fused-ssim tren GPU: ban 328dc98 chi duoc build cho kien truc GPU luc cai dat
python -c "import torch, fused_ssim; from gsplat import csrc; x = torch.rand(1, 3, 16, 16, device='cuda'); fused_ssim.fused_ssim(x, x)" \
  || die "conda env $ENV_NAME is not ready for this GPU - run scripts/setup_gsplat.sh (it rebuilds when the GPU type changes)"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

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
read -r -a SEED_LIST <<< "$SEEDS"
MCMC_EXTRA=()
if [ -n "$MCMC_DB_SET" ]; then
  MCMC_EXTRA=(--set "$MCMC_DB_SET")
fi
echo "scenes: ${SCENES[*]} | seeds: ${SEED_LIST[*]} | mcmc extra: ${MCMC_EXTRA[*]-none} | output: $OUT_ROOT"

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
  python -m zipfile -e "$ZIP" "$ARCHIVE_DIR"
  DB_ROOT="$(locate_db_root)"
fi
[ -n "$DB_ROOT" ] || die "cannot find the '$SUBDIR' folder inside $ARCHIVE_DIR"
echo "dataset root: $DB_ROOT"

# ---------------------------------------------------------------------------
FAILED=0
for seed in "${SEED_LIST[@]}"; do
  for scene in "${SCENES[@]}"; do
    src="$DB_ROOT/$scene"
    base="$OUT_ROOT/$scene/default_s$seed"
    step "2. default | $scene | seed $seed"
    if ! python -m indoor3d.train.run --mode default --scene "$src" --scene-name "$scene" --out "$base" \
         --seed "$seed" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete; then
      echo "default failed for $scene (seed $seed); skipping its mcmc run"
      FAILED=1
      continue
    fi
    step "3. mcmc (same Gaussian count) | $scene | seed $seed"
    python -m indoor3d.train.run --mode mcmc --scene "$src" --scene-name "$scene" \
      --out "$OUT_ROOT/$scene/mcmc_s$seed" --seed "$seed" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete \
      --cap-max-from "$base" "${MCMC_EXTRA[@]}" || FAILED=1
  done
done

# ---------------------------------------------------------------------------
step "4. Compare modes and check the baseline against Inria (published numbers and own B0 run)"
B0_ARGS=()
if [ -d "$B0_ROOT" ]; then
  B0_ARGS=(--b0-root "$B0_ROOT")
else
  echo "no own Inria run in $B0_ROOT (run scripts/run_baseline_deepblending.sh): only published numbers are used"
fi
python -m indoor3d.eval.compare_modes --root "$OUT_ROOT" --baseline default --modes default mcmc \
  --reference-config "$CONFIG" "${B0_ARGS[@]}" \
  --out "$REPORT_DIR/modes_db_$STAMP.json" --markdown "$REPORT_DIR/modes_db_$STAMP.md"

echo
echo "DONE. Log: $LOG"
echo "Send back: $REPORT_DIR/modes_db_$STAMP.md and .json (and $LOG if something failed)"
exit "$FAILED"
