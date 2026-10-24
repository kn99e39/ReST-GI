#!/usr/bin/env bash
# Source before any Mitsuba (cuda_*) job on LabServer63:   source server/mi_env.sh
#
# Cause of the server's Mitsuba segfault (worklogs 22/23, re-checked in worklog 43): the kernel module and libcuda
# are driver 595.71.05, but the system OptiX user-space library /lib/x86_64-linux-gnu/libnvoptix.so.1 (and
# libnvidia-rtcore) is still 570.195.03 from an earlier driver. OptiX's user-space part must match the kernel
# driver; with the mismatch Mitsuba crashes while building the OptiX acceleration structure.
# Fix without root: the matching 595.71.05 user-space files (extracted from NVIDIA's public driver package with
# --extract-only) live in ~/nvlibs/optix<version> and are put first on LD_LIBRARY_PATH.
#
# This script refuses to continue if the extracted copy does not match the running kernel driver, so a later driver
# update cannot silently bring the crash back.
KVER=$(sed -n 's/.*Kernel Module\( for [^ ]*\)\?  *\([0-9.]*\).*/\2/p' /proc/driver/nvidia/version | head -1)
NVLIB=${NVLIB:-$HOME/nvlibs/optix${KVER%%.*}}
if [ -z "$KVER" ] || [ ! -f "$NVLIB/libnvoptix.so.$KVER" ] || [ ! -f "$NVLIB/libnvidia-rtcore.so.$KVER" ]; then
    echo "mi_env: no OptiX user-space libraries for kernel driver '$KVER' in $NVLIB" >&2
    echo "mi_env: extract them: sh NVIDIA-Linux-x86_64-$KVER.run --extract-only; copy libnvoptix.so.$KVER," >&2
    echo "        libnvidia-rtcore.so.$KVER, libnvidia-gpucomp.so.$KVER, nvoptix.bin; add libnvoptix.so.1 symlink" >&2
    return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH=$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MI_OPTIX_LIB=$NVLIB
