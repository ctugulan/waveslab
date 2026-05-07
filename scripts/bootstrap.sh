#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh clone of waveslab.
#
# Recommended GPU setup on this machine:
#   ./scripts/bootstrap.sh --torch cuda --jax cuda13
#
# Other useful modes:
#   ./scripts/bootstrap.sh --torch skip --jax cuda13
#   ./scripts/bootstrap.sh --torch cpu  --jax cpu
#
# Notes:
#   - PyTorch is installed separately from pyproject.toml so the CPU/GPU wheel
#     is chosen intentionally.
#   - JAX is installed separately because plain `jax` is CPU-only on NVIDIA GPU
#     machines unless a CUDA extra is requested.
#   - SAM checkpoint files are not installed automatically. Put one of these in:
#       examples/weights/sam_vit_h_4b8939.pth
#       examples/weights/sam_vit_h_4b8939.pt

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_MODE="${TORCH_MODE:-cuda}"
JAX_MODE="${JAX_MODE:-cuda13}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/bootstrap.sh [--torch cuda|cuda126|cpu|skip] [--jax cuda13|cuda12|cpu|skip] [--python python3.12] [--venv .venv]

Examples:
  ./scripts/bootstrap.sh --torch cuda --jax cuda13
  ./scripts/bootstrap.sh --torch skip --jax cuda13
  ./scripts/bootstrap.sh --torch cpu --jax cpu
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch)
      TORCH_MODE="${2:-}"
      shift 2
      ;;
    --jax)
      JAX_MODE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

case "$TORCH_MODE" in
  cuda|cuda126|cpu|skip) ;;
  *)
    echo "ERROR: --torch must be one of: cuda, cuda126, cpu, skip"
    exit 1
    ;;
esac

case "$JAX_MODE" in
  cuda13|cuda12|cpu|skip) ;;
  *)
    echo "ERROR: --jax must be one of: cuda13, cuda12, cpu, skip"
    exit 1
    ;;
esac

echo "[1/9] Checking Python"
"$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
print(f"Python: {sys.version}")
if (major, minor) < (3, 10):
    raise SystemExit("ERROR: Python 3.10 or newer is required.")
PY

echo "[2/9] Creating virtual environment at ${VENV_DIR}"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/9] Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

uninstall_matching_distributions() {
  python - "$@" <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys

patterns = tuple(arg.lower() for arg in sys.argv[1:])
remove: list[str] = []

for dist in metadata.distributions():
    name = dist.metadata.get("Name", "")
    low = name.lower()
    if any(low == pat or low.startswith(pat) for pat in patterns):
        remove.append(name)

if remove:
    print("Uninstalling:", ", ".join(sorted(remove)))
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *sorted(remove)])
else:
    print("Nothing to uninstall.")
PY
}

echo "[4/9] Installing PyTorch mode: ${TORCH_MODE}"

if [[ "$TORCH_MODE" == "cuda" || "$TORCH_MODE" == "cuda126" ]]; then
  uninstall_matching_distributions torch torchvision torchaudio triton nvidia-
  python -m pip install --no-cache-dir \
    torch==2.8.0+cu126 \
    torchvision==0.23.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

elif [[ "$TORCH_MODE" == "cpu" ]]; then
  uninstall_matching_distributions torch torchvision torchaudio triton nvidia-
  python -m pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

elif [[ "$TORCH_MODE" == "skip" ]]; then
  echo "Skipping PyTorch install."
fi

echo "[5/9] Installing waveslab and non-JAX project dependencies"
python -m pip install -e ".[sea-ice]"

echo "[6/9] Installing JAX mode: ${JAX_MODE}"

if [[ "$JAX_MODE" == "cuda13" ]]; then
  uninstall_matching_distributions jax jaxlib jax-cuda
  python -m pip install --no-cache-dir --upgrade "jax[cuda13]"

elif [[ "$JAX_MODE" == "cuda12" ]]; then
  uninstall_matching_distributions jax jaxlib jax-cuda
  python -m pip install --no-cache-dir --upgrade "jax[cuda12]"

elif [[ "$JAX_MODE" == "cpu" ]]; then
  uninstall_matching_distributions jax jaxlib jax-cuda
  python -m pip install --no-cache-dir --upgrade "jax"

elif [[ "$JAX_MODE" == "skip" ]]; then
  echo "Skipping JAX install."
fi

echo "[7/9] Creating expected local folders"
mkdir -p examples/weights
mkdir -p outputs/sea_ice_segmentation
mkdir -p outputs/image_scattering
mkdir -p .vscode

cat > .vscode/settings.json <<EOF
{
  "python.defaultInterpreterPath": "\${workspaceFolder}/${VENV_DIR}/bin/python",
  "python.analysis.extraPaths": [
    "\${workspaceFolder}/src"
  ]
}
EOF

echo "[8/9] Running import and GPU checks"
JAX_MODE="${JAX_MODE}" TORCH_MODE="${TORCH_MODE}" python - <<'PY'
from __future__ import annotations

import importlib
import os
import shutil

required = [
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "matplotlib",
    "cameratransform",
    "segment_anything",
    "waveslab",
]

torch_mode = os.environ.get("TORCH_MODE", "cuda")
jax_mode = os.environ.get("JAX_MODE", "cuda13")

if torch_mode != "skip":
    required.extend(["torch", "torchvision"])
if jax_mode != "skip":
    required.extend(["jax", "jaxlib"])

failed = []
for name in required:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "")
        print(f"OK   {name} {version}".rstrip())
    except Exception as exc:
        failed.append((name, exc))
        print(f"MISS {name}: {exc}")

if failed:
    print("\nERROR: Some imports failed.")
    for name, exc in failed:
        print(f"  - {name}: {exc}")
    raise SystemExit(1)

if torch_mode != "skip":
    import torch

    print(f"\nTorch version: {torch.__version__}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch_mode.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("ERROR: requested CUDA PyTorch, but torch.cuda.is_available() is False.")
    if torch.cuda.is_available():
        print(f"Torch CUDA device: {torch.cuda.get_device_name(0)}")

if jax_mode != "skip":
    if jax_mode.startswith("cuda"):
        os.environ.setdefault("JAX_PLATFORMS", "cuda")

    import jax
    import jaxlib

    print(f"\nJAX version: {jax.__version__}")
    print(f"JAXLIB version: {jaxlib.__version__}")
    print(f"JAX default backend: {jax.default_backend()}")
    print("JAX devices:")
    for dev in jax.devices():
        print(f"  - {dev}")

    gpu_devices = [
        dev for dev in jax.devices()
        if getattr(dev, "platform", "").lower() in {"gpu", "cuda"}
    ]

    if jax_mode.startswith("cuda") and not gpu_devices:
        nvidia_hint = ""
        if shutil.which("nvidia-smi"):
            nvidia_hint = "\nNVIDIA driver appears to be present, but JAX still did not find a CUDA device."
        raise SystemExit(
            "ERROR: requested CUDA JAX, but no JAX GPU device was found."
            + nvidia_hint
        )
PY

echo "[9/9] Checking SAM checkpoint"
if [[ -f "examples/weights/sam_vit_h_4b8939.pth" ]]; then
  echo "OK   examples/weights/sam_vit_h_4b8939.pth"
elif [[ -f "examples/weights/sam_vit_h_4b8939.pt" ]]; then
  echo "OK   examples/weights/sam_vit_h_4b8939.pt"
  echo "NOTE: your script default expects .pth, so either pass --sam-weights or copy this file:"
  echo "      cp examples/weights/sam_vit_h_4b8939.pt examples/weights/sam_vit_h_4b8939.pth"
else
  echo "WARN missing SAM checkpoint."
  echo "     Put the SAM ViT-H checkpoint here:"
  echo "       examples/weights/sam_vit_h_4b8939.pth"
fi

echo
echo "Bootstrap complete."
echo
cat <<'EOF'
Next steps:
  source .venv/bin/activate

  python examples/segment_sea_ice_fair_compare.py \
    --image "examples/2017SeaIceImage/2017-07-04/17-07-04 10-00-53.bmp" \
    --sam-weights "examples/weights/sam_vit_h_4b8939.pth" \
    --out outputs/sea_ice_segmentation \
    --selected-method sam_auto \
    --save-diagnostics

Then run the image scattering comparison with strict JAX GPU mode:
  python examples/run_image_scattering_comparison.py --backend cuda ...
EOF