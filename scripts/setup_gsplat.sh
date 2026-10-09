#!/usr/bin/env bash
# setup_gsplat.sh - Cai env gs-gsplat cho hai che do huan luyen (Default / MCMC) tren pod Runpod.
# Moi thu nam trong /workspace (network volume) nen con nguyen sau khi Stop pod.
# Chay lai an toan: buoc nao da xong se duoc bo qua.
#
# Cach chay (tu thu muc goc cua repo tren pod):
#   bash scripts/setup_gsplat.sh
#
# Bien tuy chon:
#   WS=/workspace          thu muc goc tren network volume
#   ENV_NAME=gs-gsplat     ten conda env
#   MAX_JOBS=8             so tien trinh bien dich song song
#   FORCE_REBUILD=1        build lai gsplat va fused-ssim
#   ALLOW_PROVISIONAL=1    cho phep chay khi phien ban trong file conf con "provisional"
#
# Phien ban ghim doc tu environment/gs-gsplat.conf va environment/gs-gsplat.lock.txt.
# Xem docs/environment.md (muc 4.3) va docs/design/training_modes.md.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${CONF:-$REPO_DIR/environment/gs-gsplat.conf}"
[ -r "$CONF" ] || { echo "ERROR: missing $CONF" >&2; exit 1; }
# shellcheck source=/dev/null
. "$CONF"
for v in PYTHON_VERSION TORCH_VERSION TORCHVISION_VERSION TORCH_CUDA TORCH_CUDA_ARCH_LIST \
         GSPLAT_REPO GSPLAT_COMMIT FUSED_SSIM_REPO FUSED_SSIM_COMMIT LOCK_FILE; do
  [ -n "${!v:-}" ] || { echo "ERROR: $v is missing in $CONF" >&2; exit 1; }
done
LOCK="$REPO_DIR/$LOCK_FILE"
[ -r "$LOCK" ] || { echo "ERROR: missing $LOCK" >&2; exit 1; }

WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
CONDA_DIR="$WS/miniforge3"
CODE_DIR="$WS/code/gsplat"
LOG_DIR="$WS/logs"
REPORT_DIR="$WS/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$WS/code"
LOG="$LOG_DIR/setup_gsplat_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
die()  { echo "ERROR: $*"; exit 1; }
warn() { echo "WARN: $*"; }
# ver_ge A B -> true if version A >= version B
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]; }
# cu128 -> 12.8, cu118 -> 11.8, cu130 -> 13.0
cuda_from_tag() { local d="${1#cu}"; echo "${d:0:${#d}-1}.${d: -1}"; }

REQ_CUDA="$(cuda_from_tag "$TORCH_CUDA")"
REQ_MAJOR="${REQ_CUDA%%.*}"

# ---------------------------------------------------------------------------
step "0. Preflight"
if [ "${ENV_STATUS:-}" = "provisional" ] && [ "${ALLOW_PROVISIONAL:-0}" != "1" ]; then
  die "versions in $CONF are still provisional. Confirm them from the env_probe.sh report first (or rerun with ALLOW_PROVISIONAL=1)."
fi
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (no GPU visible)"
command -v git >/dev/null 2>&1 || die "git not found"
command -v curl >/dev/null 2>&1 || die "curl not found"

INRIA_CONF="$REPO_DIR/environment/gs-inria.conf"
if [ -r "$INRIA_CONF" ]; then
  # shellcheck source=/dev/null
  inria_torch="$(. "$INRIA_CONF" && echo "$TORCH_VERSION+$TORCH_CUDA")"
  [ "$inria_torch" = "$TORCH_VERSION+$TORCH_CUDA" ] \
    || warn "PyTorch differs from gs-inria.conf ($inria_torch vs $TORCH_VERSION+$TORCH_CUDA): record it when comparing B0 and B1"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)"
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)"
echo "GPU: ${GPU_NAME:-?} | compute capability: ${GPU_CC:-?} | VRAM: ${GPU_MIB:-?} MiB"
echo "Driver supports CUDA: ${DRIVER_CUDA:-?} | requested PyTorch: $TORCH_VERSION+$TORCH_CUDA (CUDA $REQ_CUDA)"

[ -n "$DRIVER_CUDA" ] || die "cannot read the driver CUDA version from nvidia-smi"
ver_ge "$DRIVER_CUDA" "$REQ_CUDA" \
  || die "driver supports CUDA $DRIVER_CUDA < $REQ_CUDA. Choose an older TORCH_CUDA in $CONF."
case "${GPU_CC:-}" in
  1[0-9].*) ver_ge "$REQ_CUDA" "12.8" || die "Blackwell GPU (cc $GPU_CC) needs CUDA >= 12.8" ;;
esac
case ";$TORCH_CUDA_ARCH_LIST;" in
  *";${GPU_CC:-none};"*) ;;
  *) warn "GPU compute capability ${GPU_CC:-?} is not in TORCH_CUDA_ARCH_LIST ($TORCH_CUDA_ARCH_LIST)" ;;
esac
if [ "${GPU_MIB:-0}" -lt 23000 ] 2>/dev/null; then
  warn "VRAM < 24 GB: large scenes may run out of memory"
fi
if [ ! -d "$WS" ] || [ ! -w "$WS" ]; then
  die "$WS is not a writable directory"
fi
FREE_GB="$(df -BG --output=avail "$WS" | tail -n 1 | tr -dc '0-9')"
[ "${FREE_GB:-0}" -ge 20 ] || die "need >= 20 GB free in $WS (have ${FREE_GB:-?} GB)"
for url in https://github.com https://download.pytorch.org/whl/ https://pypi.org/simple/ \
           https://conda.anaconda.org/conda-forge/; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$url" || true)"
  case "$code" in
    2??|3??) echo "reachable: $url ($code)" ;;
    *) die "cannot reach $url (HTTP ${code:-none})" ;;
  esac
done

# ---------------------------------------------------------------------------
step "1. Miniforge (conda) in $CONDA_DIR"
if [ -x "$CONDA_DIR/bin/conda" ]; then
  echo "already installed: $("$CONDA_DIR/bin/conda" --version)"
else
  tmpdir="$(mktemp -d)"
  curl -fsSL -o "$tmpdir/miniforge.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$tmpdir/miniforge.sh" -b -p "$CONDA_DIR"
  rm -rf "$tmpdir"
fi
# shellcheck source=/dev/null
. "$CONDA_DIR/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
step "2. Conda env $ENV_NAME (Python $PYTHON_VERSION)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env already exists"
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi
set +u
conda activate "$ENV_NAME"
set -u
echo "python: $(command -v python) ($(python --version 2>&1))"

# ---------------------------------------------------------------------------
step "3. PyTorch $TORCH_VERSION ($TORCH_CUDA)"
if python - "$TORCH_VERSION" "$REQ_CUDA" >/dev/null 2>&1 <<'PY'
import sys
import torch
want_ver, want_cuda = sys.argv[1], sys.argv[2]
ok = torch.__version__.split("+")[0] == want_ver and torch.version.cuda == want_cuda
sys.exit(0 if ok else 1)
PY
then
  echo "already installed"
else
  pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
PY

# ---------------------------------------------------------------------------
step "4. CUDA compiler (nvcc) for building extensions"
NVCC=""
if [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
  NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
fi
[ -n "$NVCC" ] || die "nvcc not found. Send the env_probe.sh report so a toolkit install step can be added."
NVCC_VER="$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
[ "${NVCC_VER%%.*}" = "$REQ_MAJOR" ] \
  || die "nvcc $NVCC_VER and PyTorch CUDA $REQ_CUDA have different major versions"
CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
export CUDA_HOME
echo "nvcc: $NVCC (release $NVCC_VER) | CUDA_HOME=$CUDA_HOME | gcc: $(gcc -dumpfullversion 2>/dev/null || echo unknown)"
if [ "$NVCC_VER" != "$REQ_CUDA" ]; then
  warn "nvcc $NVCC_VER differs from PyTorch CUDA $REQ_CUDA in the minor version (allowed by PyTorch, recorded in the report)"
fi

# ---------------------------------------------------------------------------
step "5. gsplat @ ${GSPLAT_COMMIT:0:7}"
if [ ! -d "$CODE_DIR/.git" ]; then
  git clone "$GSPLAT_REPO" "$CODE_DIR"
fi
if [ "$(git -C "$CODE_DIR" rev-parse HEAD)" != "$GSPLAT_COMMIT" ]; then
  git -C "$CODE_DIR" fetch origin
  git -C "$CODE_DIR" checkout --detach "$GSPLAT_COMMIT"
fi
git -C "$CODE_DIR" submodule update --init --recursive
[ "$(git -C "$CODE_DIR" rev-parse HEAD)" = "$GSPLAT_COMMIT" ] || die "gsplat is not at the pinned commit"
git -C "$CODE_DIR" submodule status
if git -C "$CODE_DIR" submodule status | grep -q '^[-+U]'; then
  die "a gsplat submodule (glm) is missing or not at the pinned commit"
fi
if [ -n "$(git -C "$CODE_DIR" status --porcelain --untracked-files=no)" ]; then
  die "$CODE_DIR has local modifications; the pipeline expects unmodified gsplat sources"
fi

# ---------------------------------------------------------------------------
step "6. Python dependencies ($LOCK_FILE)"
CONSTRAINTS="$(mktemp)"
pip freeze | grep -iE '^(torch|torchvision)==' > "$CONSTRAINTS"
echo "keeping: $(tr '\n' ' ' < "$CONSTRAINTS")"
pip install -c "$CONSTRAINTS" -r "$LOCK"
rm -f "$CONSTRAINTS"
pip check || die "pip reports conflicting dependencies (see above)"

# ---------------------------------------------------------------------------
step "7. Build gsplat and fused-ssim (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-8}"
MARKER="$CONDA_DIR/envs/$ENV_NAME/.indoor3d_gsplat_build"
# fused-ssim @ 328dc98 tu do kien truc cua GPU dang gan va bo qua TORCH_CUDA_ARCH_LIST,
# nen doi loai GPU (compute capability khac) thi phai build lai: ghi GPU_CC vao marker.
WANT_BUILD="gsplat=$GSPLAT_COMMIT fused_ssim=$FUSED_SSIM_COMMIT arch=$TORCH_CUDA_ARCH_LIST gpu_cc=${GPU_CC:-?} torch=$TORCH_VERSION+$TORCH_CUDA"
if [ "${FORCE_REBUILD:-0}" != "1" ] && [ "$(cat "$MARKER" 2>/dev/null || true)" = "$WANT_BUILD" ] \
   && python -c "from gsplat import csrc; import fused_ssim" >/dev/null 2>&1; then
  echo "already built: $WANT_BUILD (set FORCE_REBUILD=1 to rebuild)"
else
  rm -f "$MARKER"
  echo "--- building gsplat (several minutes) ---"
  pip install --no-build-isolation --no-deps --force-reinstall "$CODE_DIR"
  echo "--- building fused-ssim ---"
  pip install --no-build-isolation --no-deps --force-reinstall "git+$FUSED_SSIM_REPO@$FUSED_SSIM_COMMIT"
  python -c "from gsplat import csrc; import fused_ssim" \
    || die "gsplat.csrc or fused_ssim is not importable after the build"
  echo "$WANT_BUILD" > "$MARKER"
fi

# ---------------------------------------------------------------------------
step "8. Smoke test"
python "$REPO_DIR/scripts/smoke_test_gsplat.py" --examples-dir "$CODE_DIR/examples"

# ---------------------------------------------------------------------------
step "9. Record environment"
SUMMARY="$REPORT_DIR/setup_gsplat_$STAMP.txt"
{
  echo "date: $(date -Is)"
  echo "gpu: ${GPU_NAME:-?} (cc ${GPU_CC:-?}, ${GPU_MIB:-?} MiB) | driver CUDA ${DRIVER_CUDA:-?}"
  echo "conda: $(conda --version)"
  echo "env: $ENV_NAME ($(python --version 2>&1))"
  python -c 'import torch, torchvision; print("torch", torch.__version__, "| cuda", torch.version.cuda, "| torchvision", torchvision.__version__)'
  echo "nvcc: $NVCC_VER ($NVCC)"
  echo "gcc: $(gcc -dumpfullversion 2>/dev/null || echo unknown)"
  echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
  echo "gsplat commit: $(git -C "$CODE_DIR" rev-parse HEAD)"
  git -C "$CODE_DIR" submodule status
  echo "fused-ssim commit: $FUSED_SSIM_COMMIT"
  echo "lock file: $LOCK_FILE (sha256 $(sha256sum "$LOCK" | cut -c1-16))"
  echo "conf: $CONF (ENV_STATUS=${ENV_STATUS:-unset})"
} | tee "$SUMMARY"
pip freeze > "$REPORT_DIR/pip_freeze_${ENV_NAME}_$STAMP.txt"
conda env export -n "$ENV_NAME" --no-builds > "$REPORT_DIR/conda_env_${ENV_NAME}_$STAMP.yml"

echo
echo "DONE. Log: $LOG"
echo "Send back: $SUMMARY and the pip_freeze / conda_env files in $REPORT_DIR"
echo "Activate later with:  . $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME"
