#!/usr/bin/env bash
# env_probe.sh - Ghi lai moi truong thuc te cua pod GPU.
# Chi DOC thong tin: khong cai dat, khong sua file he thong.
#
# Cach chay tren pod:
#   bash /workspace/env_probe.sh
# Ket qua in ra man hinh va luu vao /workspace/reports/env_<thoi_gian>.txt
# Doi thu muc luu:  OUT_DIR=/duong/dan bash env_probe.sh
#
# Script co y KHONG in toan bo bien moi truong de tranh lo API key / bi mat cua pod.

set -u

SCRIPT_VERSION="1"
OUT_DIR="${OUT_DIR:-/workspace/reports}"
mkdir -p "$OUT_DIR" || { echo "Cannot create $OUT_DIR" >&2; exit 1; }
OUT="$OUT_DIR/env_$(date +%Y%m%d_%H%M%S).txt"

PY="$(command -v python3 || command -v python || true)"

section() { printf '\n===== %s =====\n' "$1"; }

# Run a command if it exists; never abort the whole report on failure.
run() {
  echo "\$ $*"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 || echo "(exit code $?)"
  else
    echo "(not found: $1)"
  fi
}

# Same as run, but keep only the first line of output (version strings).
first() {
  echo "\$ $*"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 | head -n 1
  else
    echo "(not found: $1)"
  fi
}

{
  echo "env_probe.sh version $SCRIPT_VERSION"

  section "Time & host"
  date -Is 2>/dev/null || date
  run uname -srm
  if [ -r /etc/os-release ]; then
    grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
  fi
  # Only selected, non-secret variables.
  for v in RUNPOD_DC_ID RUNPOD_GPU_COUNT CUDA_VERSION; do
    printf '%s=%s\n' "$v" "${!v:-<unset>}"
  done

  section "GPU & driver"
  run nvidia-smi
  run nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

  section "CUDA toolkit inside container"
  if command -v nvcc >/dev/null 2>&1; then
    run nvcc --version
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    echo "(nvcc not on PATH, found /usr/local/cuda/bin/nvcc)"
    /usr/local/cuda/bin/nvcc --version 2>&1
  else
    echo "(nvcc not found)"
  fi
  ls -d /usr/local/cuda* 2>/dev/null || echo "(no /usr/local/cuda*)"

  section "Compilers & build tools"
  first gcc --version
  first g++ --version
  first cmake --version
  first ninja --version
  first conda --version
  first git --version

  section "Python & PyTorch"
  if [ -n "$PY" ]; then
    echo "python executable: $PY"
    "$PY" - <<'PYEOF' 2>&1
import platform
import sys

print("python version:", platform.python_version())
try:
    import torch
except Exception as e:  # torch missing or broken
    print("torch import failed:", repr(e))
    sys.exit(0)

print("torch:", torch.__version__)
print("torch built with CUDA:", torch.version.cuda)
try:
    print("cuDNN:", torch.backends.cudnn.version())
except Exception as e:
    print("cuDNN: n/a", repr(e))

ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    print("device:", p.name)
    print("VRAM (GiB): %.1f" % (p.total_memory / 2**30))
    print("compute capability: %d.%d" % torch.cuda.get_device_capability(i))
    print("compiled arch list:", torch.cuda.get_arch_list())
    try:
        a = torch.randn(4096, 4096, device="cuda")
        _ = (a @ a).sum().item()
        torch.cuda.synchronize()
        print("GPU matmul test: OK")
    except Exception as e:
        print("GPU matmul test FAILED:", repr(e))
PYEOF
  else
    echo "(python not found)"
  fi

  section "Selected Python packages"
  if [ -n "$PY" ]; then
    "$PY" -m pip list 2>/dev/null \
      | grep -iE '^(torch|torchvision|torchaudio|numpy|jupyterlab|opencv[-_a-z]*|plyfile|ninja)[[:space:]]' \
      || echo "(none of the selected packages found)"
  fi

  section "System tools"
  for t in colmap ffmpeg magick convert git-lfs tmux rsync wget curl unzip; do
    printf '%-9s ' "$t:"
    command -v "$t" || echo "not found"
  done

  section "CPU / RAM / disk"
  echo "CPU cores (nproc): $(nproc 2>/dev/null || echo n/a)"
  run free -h
  echo "\$ df -h / /workspace /dev/shm"
  df -h / /workspace /dev/shm 2>&1
  echo "--- /workspace mount (mountpoint, filesystem type) ---"
  awk '$2=="/workspace" {print $2, $3}' /proc/mounts 2>/dev/null | grep . \
    || echo "(no separate mount for /workspace)"

  section "Internet access (needed for code and datasets)"
  if command -v curl >/dev/null 2>&1; then
    for url in https://github.com https://pypi.org https://repo-sam.inria.fr; do
      code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" 2>/dev/null)" || code="fail"
      echo "$url -> $code"
    done
  else
    echo "(curl not found)"
  fi

  section "SUMMARY (automatic checks)"
  vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
  case "$vram_mib" in
    ''|*[!0-9]*) echo "[FAIL] GPU not visible to nvidia-smi" ;;
    *) if [ "$vram_mib" -ge 23000 ]; then
         echo "[PASS] VRAM ${vram_mib} MiB (>= 24 GB class)"
       else
         echo "[WARN] VRAM ${vram_mib} MiB (< 24 GB, paper-quality 3DGS may OOM)"
       fi ;;
  esac
  if [ -n "$PY" ] && "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "[PASS] PyTorch sees the GPU"
  else
    echo "[FAIL] PyTorch does not see the GPU (or torch missing)"
  fi
  if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
    echo "[PASS] nvcc available (needed to build 3DGS CUDA extensions)"
  else
    echo "[WARN] nvcc missing (will be handled when installing 3DGS)"
  fi
  if awk '$2=="/workspace" {found=1} END {exit !found}' /proc/mounts 2>/dev/null; then
    echo "[PASS] /workspace is a separate mount (volume disk or network volume - confirm type in Runpod console)"
  else
    echo "[WARN] /workspace is not a separate mount (data may not persist)"
  fi
  free_gb="$(df -BG --output=avail /workspace 2>/dev/null | tail -n 1 | tr -dc '0-9')"
  if [ -n "$free_gb" ] && [ "$free_gb" -ge 50 ]; then
    echo "[PASS] /workspace free space ${free_gb} GB (>= 50 GB)"
  else
    echo "[WARN] /workspace free space ${free_gb:-unknown} GB (< 50 GB or unknown)"
  fi
} 2>&1 | tee "$OUT"

echo
echo "Report saved to: $OUT"
