#!/usr/bin/env bash
# WL109B Amendment 3: run one command of the frozen WL109B stack inside a Docker container on LabServer70.
# usage: ctr109b.sh NAME CMD ARGS...
# Why: LabServer70 runs CentOS 7 (glibc 2.17); the frozen Dr.Jit 1.3.1 / Mitsuba 3.8.0 binaries need glibc >= 2.27. The
# container (Debian 12, glibc 2.36, image pinned by ID) supplies only the C library userland; the Python interpreter, venv,
# Mitsuba build, code and data are the same files, bind-mounted at their own paths. GPU: the RTX 3090 only (UUID).
# NVIDIA container toolkit 1.11 predates driver R535 and does not inject three files OptiX 8 needs on driver 550.54.15
# (nvoptix.bin, libnvidia-gpucomp, libnvidia-nvvm.so.4); they are bind-mounted read-only from the host driver install.
set -eu
name=$1; shift
IMAGE=sha256:6fd6ebbd05fc4cf5dc3a789c47e42f4d787cb9c90fe5d29835875a934680271d   # python:3.10-bookworm (python@sha256:f72a8680...)
GPU3090=GPU-ab3bc073-096c-6d0b-a304-13cef177a843
L=/usr/lib/x86_64-linux-gnu
exec docker run --rm --init --name "$name" --hostname lab70-wl109b \
  --gpus "\"device=$GPU3090\"" -e NVIDIA_DRIVER_CAPABILITIES=all \
  -u 1014:1014 -v /home/nd:/home/nd -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v /usr/lib64/libnvidia-gpucomp.so.550.54.15:$L/libnvidia-gpucomp.so.550.54.15:ro \
  -v /usr/lib64/libnvidia-nvvm.so.550.54.15:$L/libnvidia-nvvm.so.4:ro \
  -e HOME=/home/nd -e DYN101_MI_DIR=/home/nd/rest_dyn101/external/mi_3.8.0 -e DYN101_MI_VERSION=3.8.0 \
  -e LD_LIBRARY_PATH=/home/nd/rest_sss101a/results/ecro_estimator109b/runtime70/lib \
  -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" -e TMPDIR="${TMPDIR:-/tmp}" -e WL109B_HOST=LabServer70 \
  -w "$PWD" "$IMAGE" "$@"
