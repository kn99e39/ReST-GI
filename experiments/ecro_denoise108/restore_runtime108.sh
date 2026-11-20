#!/usr/bin/env bash
# Restore only the WL108 runtime and preserve the archived environment directory.
set -eu -o pipefail
cd "$HOME/rest_sss101a"
runtime="$PWD/results/ecro_denoise108/runtime"
mkdir -p "$runtime"
cd "$runtime"

fetch_verified() {
  local url=$1 name=$2 checksum=$3
  curl --fail --location --retry 2 --silent --show-error "$url" -o "$name"
  curl --fail --location --retry 2 --silent --show-error "$checksum" -o "$name.sha256"
  local digest
  digest=$(grep -Eo '[[:xdigit:]]{64}' "$name.sha256" | head -1)
  printf '%s  %s\n' "$digest" "$name" | sha256sum -c -
}

python_release='https://github.com/astral-sh/python-build-standalone/releases/download/20260901'
python_file='cpython-3.10.21+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz'
python_url="$python_release/${python_file/+/%2B}"
if [[ ! -x python/bin/python3.10 ]]; then
  fetch_verified "$python_url" "$python_file" "$python_url.sha256"
  tar -xf "$python_file"
fi
py="$runtime/python/bin/python3.10"
"$py" --version
"$py" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.3.1+cu121' 'triton==2.3.1'
"$py" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple \
  'numpy==1.26.4' 'scipy==1.14.1' 'mitsuba==3.8.0' 'drjit==1.3.1' 'matplotlib==3.10.9' 'pillow==12.3.0'

driver_url='https://download.nvidia.com/XFree86/Linux-x86_64/595.71.05'
driver='NVIDIA-Linux-x86_64-595.71.05-no-compat32.run'
mkdir -p drivers
if [[ ! -f drivers/libnvoptix.so.595.71.05 ]]; then
  fetch_verified "$driver_url/$driver" "$driver" "$driver_url/$driver.sha256sum"
  sh "$driver" --extract-only --target "$runtime/nvidia_extracted"
  for lib in libnvoptix.so.595.71.05 libnvidia-gpucomp.so.595.71.05 libcuda.so.595.71.05 libnvidia-ptxjitcompiler.so.595.71.05; do
    cp "nvidia_extracted/$lib" drivers/
    ln -s "$lib" "drivers/${lib%.595.71.05}.1"
  done
  ln -s libnvidia-gpucomp.so.595.71.05 drivers/libnvidia-gpucomp.so
fi
printf '%s  %s\n' '69be01c2e2794dbf775b9664a9c641c2db49efe5383a7a56b5f29aaa0de07fc4' \
  'drivers/libnvoptix.so.595.71.05' | sha256sum -c -
export LD_LIBRARY_PATH="$runtime/drivers:${LD_LIBRARY_PATH:-}"
"$py" - <<'PY'
import sys
import drjit
import mitsuba
import numpy
import torch
import triton

assert sys.version_info[:3] == (3, 10, 21)
assert (torch.__version__, triton.__version__, numpy.__version__) == ('2.3.1+cu121', '2.3.1', '1.26.4')
assert (mitsuba.__version__, drjit.__version__) == ('3.8.0', '1.3.1')
assert torch.cuda.is_available()
print("Recovered runtime", torch.cuda.get_device_name(0))
PY
