#!/usr/bin/env bash
# WL111A copy of WL111 ctr_r111.sh (GPU selectable through WL111A_GPU). WL111: run one command of the frozen stack inside a Docker container on LabServer70 (copy of WL109B ctr109b.sh with this
# batch's name and environment). usage: ctr_r111.sh NAME CMD ARGS...
# The container supplies only the glibc userland (Debian 12; the host is CentOS 7); interpreter, venv, Mitsuba build, code and
# data are host files bind-mounted at their own paths. GPU: the RTX 3090 only (UUID). The three OptiX 8 driver files the
# NVIDIA container toolkit 1.11 does not inject are bind-mounted read-only from the host driver install.
set -eu
name=$1; shift
IMAGE=sha256:6fd6ebbd05fc4cf5dc3a789c47e42f4d787cb9c90fe5d29835875a934680271d   # python:3.10-bookworm
GPU3090=${WL111A_GPU:-GPU-ab3bc073-096c-6d0b-a304-13cef177a843}   # WL111A: device selectable (default RTX 3090)
L=/usr/lib/x86_64-linux-gnu
R=/home/nd/rest_sss101a
# WL111 Amendment 1 (owner, 2026-11-26): no CPU cap by default; WL111_CPUS=N sets one explicitly.
exec docker run --rm --init --name "$name" --hostname lab70-wl111a ${WL111_CPUS:+--cpus "$WL111_CPUS"} \
  --gpus "\"device=$GPU3090\"" -e NVIDIA_DRIVER_CAPABILITIES=all \
  -u 1014:1014 -v /home/nd:/home/nd -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v /usr/lib64/libnvidia-gpucomp.so.550.54.15:$L/libnvidia-gpucomp.so.550.54.15:ro \
  -v /usr/lib64/libnvidia-nvvm.so.550.54.15:$L/libnvidia-nvvm.so.4:ro \
  -e HOME=/home/nd -e DYN101_MI_DIR=/home/nd/rest_dyn101/external/mi_3.8.0 -e DYN101_MI_VERSION=3.8.0 -e FREEZE=1 \
  -e LD_LIBRARY_PATH=$R/results/ecro_denoise108/runtime/lib \
  -e RECON111_ROOT=$R/results/ecro_recon111 -e RECON111_GUARDED=1 \
  -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e MPLCONFIGDIR=$R/results/ecro_recon111/mpl -e TMPDIR=$R/results/ecro_recon111/tmp -e TRITON_CACHE_DIR=$R/results/ecro_recon111/triton \
  -w "$PWD" "$IMAGE" "$@"
