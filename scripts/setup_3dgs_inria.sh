#!/usr/bin/env bash
# setup_3dgs_inria.sh - Cai moi truong cho 3DGS goc cua Inria (baseline B0) tren pod Runpod.
# Moi thu nam trong /workspace (network volume) nen con nguyen sau khi Stop pod.
# Chay lai an toan: buoc nao da xong se duoc bo qua.
#
# Cach chay (tu thu muc goc cua repo tren pod):
#   bash scripts/setup_3dgs_inria.sh
#
# Bien tuy chon:
#   WS=/workspace          thu muc goc tren network volume
#   ENV_NAME=gs-inria      ten conda env
#   FORCE_REBUILD=1        build lai CUDA extension du da import duoc
#   ALLOW_PROVISIONAL=1    cho phep chay khi phien ban trong file conf con "provisional"
#
# Phien ban ghim doc tu environment/gs-inria.conf. Xem docs/environment.md.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${CONF:-$REPO_DIR/environment/gs-inria.conf}"
[ -r "$CONF" ] || { echo "ERROR: missing $CONF" >&2; exit 1; }
# shellcheck source=/dev/null
. "$CONF"
for v in PYTHON_VERSION TORCH_VERSION TORCHVISION_VERSION TORCH_CUDA \
         TORCH_CUDA_ARCH_LIST INRIA_REPO INRIA_COMMIT; do
  [ -n "${!v:-}" ] || { echo "ERROR: $v is missing in $CONF" >&2; exit 1; }
done

SIMPLE_KNN_MIRROR="${SIMPLE_KNN_MIRROR:-https://github.com/camenduru/simple-knn.git}"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-inria}"
CONDA_DIR="$WS/miniforge3"
CODE_DIR="$WS/code/gaussian-splatting"
LOG_DIR="$WS/logs"
REPORT_DIR="$WS/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$WS/code"
LOG="$LOG_DIR/setup_3dgs_inria_$STAMP.log"
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
  warn "VRAM < 24 GB: paper-quality training may run out of memory"
fi
if [ ! -d "$WS" ] || [ ! -w "$WS" ]; then
  die "$WS is not a writable directory"
fi
FREE_GB="$(df -BG --output=avail "$WS" | tail -n 1 | tr -dc '0-9')"
[ "${FREE_GB:-0}" -ge 30 ] || die "need >= 30 GB free in $WS (have ${FREE_GB:-?} GB)"
for url in https://github.com https://download.pytorch.org/whl/ https://conda.anaconda.org/conda-forge/; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$url" || true)"
  case "$code" in
    2??|3??) echo "reachable: $url ($code)" ;;
    *) die "cannot reach $url (HTTP ${code:-none})" ;;
  esac
done
# gitlab.inria.fr hosts the simple-knn submodule; if it is blocked, step 5 falls back
# to a GitHub mirror and checks out the same pinned commit (identical content).
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://gitlab.inria.fr/bkerbl/simple-knn || true)"
case "$code" in
  2??|3??) echo "reachable: gitlab.inria.fr ($code)" ;;
  *) warn "gitlab.inria.fr not reachable (HTTP ${code:-none}); will use the GitHub mirror for simple-knn" ;;
esac

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
step "5. Inria gaussian-splatting @ ${INRIA_COMMIT:0:7}"
if [ ! -d "$CODE_DIR/.git" ]; then
  git clone "$INRIA_REPO" "$CODE_DIR"
fi
if [ "$(git -C "$CODE_DIR" rev-parse HEAD)" != "$INRIA_COMMIT" ]; then
  git -C "$CODE_DIR" fetch origin
  git -C "$CODE_DIR" checkout --detach "$INRIA_COMMIT"
fi
# Only the submodules needed for training; SIBR_viewers (desktop viewer) is skipped.
git -C "$CODE_DIR" submodule update --init --recursive \
  submodules/diff-gaussian-rasterization submodules/fused-ssim
if ! git -C "$CODE_DIR" submodule update --init submodules/simple-knn; then
  # The superproject pins the exact commit SHA, so the mirror yields identical code.
  warn "simple-knn could not be fetched from gitlab.inria.fr; using GitHub mirror $SIMPLE_KNN_MIRROR"
  git -C "$CODE_DIR" config "submodule.submodules/simple-knn.url" "$SIMPLE_KNN_MIRROR"
  git -C "$CODE_DIR" submodule update --init submodules/simple-knn
fi
[ "$(git -C "$CODE_DIR" rev-parse HEAD)" = "$INRIA_COMMIT" ] || die "Inria repo is not at the pinned commit"
NEEDED_SUBMODULES=(submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim)
git -C "$CODE_DIR" submodule status "${NEEDED_SUBMODULES[@]}"
if git -C "$CODE_DIR" submodule status "${NEEDED_SUBMODULES[@]}" | grep -q '^[-+U]'; then
  die "a required submodule is missing or not at the commit pinned by the Inria repo"
fi

# ---------------------------------------------------------------------------
step "6. Python dependencies"
# opencv-python-headless instead of opencv-python: servers usually lack libGL.
pip install plyfile tqdm opencv-python-headless joblib ninja

# ---------------------------------------------------------------------------
step "7. Build CUDA extensions (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
export TORCH_CUDA_ARCH_LIST
if [ "${FORCE_REBUILD:-0}" != "1" ] \
   && python -c "import diff_gaussian_rasterization, simple_knn._C, fused_ssim" >/dev/null 2>&1; then
  echo "extensions already importable (set FORCE_REBUILD=1 to rebuild)"
else
  for ext in diff-gaussian-rasterization simple-knn fused-ssim; do
    echo "--- building $ext ---"
    pip install --no-build-isolation --no-deps --force-reinstall "$CODE_DIR/submodules/$ext"
  done
fi

# ---------------------------------------------------------------------------
step "8. Smoke test"
python "$REPO_DIR/scripts/smoke_test_3dgs.py"

# ---------------------------------------------------------------------------
step "9. Record environment"
SUMMARY="$REPORT_DIR/setup_3dgs_inria_$STAMP.txt"
{
  echo "date: $(date -Is)"
  echo "gpu: ${GPU_NAME:-?} (cc ${GPU_CC:-?}, ${GPU_MIB:-?} MiB) | driver CUDA ${DRIVER_CUDA:-?}"
  echo "conda: $(conda --version)"
  echo "env: $ENV_NAME ($(python --version 2>&1))"
  python -c 'import torch, torchvision; print("torch", torch.__version__, "| cuda", torch.version.cuda, "| torchvision", torchvision.__version__)'
  echo "nvcc: $NVCC_VER ($NVCC)"
  echo "gcc: $(gcc -dumpfullversion 2>/dev/null || echo unknown)"
  echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
  echo "inria commit: $(git -C "$CODE_DIR" rev-parse HEAD)"
  git -C "$CODE_DIR" submodule status
  echo "conf: $CONF (ENV_STATUS=${ENV_STATUS:-unset})"
} | tee "$SUMMARY"
pip freeze > "$REPORT_DIR/pip_freeze_${ENV_NAME}_$STAMP.txt"
conda env export -n "$ENV_NAME" --no-builds > "$REPORT_DIR/conda_env_${ENV_NAME}_$STAMP.yml"

echo
echo "DONE. Log: $LOG"
echo "Send back: $SUMMARY and the pip_freeze / conda_env files in $REPORT_DIR"
echo "Activate later with:  . $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME"
