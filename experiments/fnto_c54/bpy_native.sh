#!/usr/bin/env bash
# Run Blender-as-a-module (bpy 3.5.1) on LabServer69 WITHOUT Docker: the same venv python and the same bpy wheel that the
# restgi/bpy-runtime:22.04 container runs, but loaded by the image's own Ubuntu 22.04 dynamic loader and libraries, extracted to
# /home (~/rest_wl54/bpyroot, 235 MB) -- bpy needs glibc 2.35 / GLIBCXX_3.4.30, the CentOS 8 host has glibc 2.28. Nothing touches /.
#   bpy_native.sh <script.py> [args...]          (cwd: experiments/fnto_c54 or any)
# Rebuild bpyroot:  docker run --rm -u 1013:1013 -v ~/rest_wl54/bpyroot:/out restgi/bpy-runtime:22.04 \
#                     bash -c "cp -a /usr/lib/x86_64-linux-gnu /out/lib && cp -aL /lib64/ld-linux-x86-64.so.2 /out/"
R=$HOME/rest_wl54/bpyroot
V=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv
PY=$(readlink -f $V/bin/python)                         # uv cpython 3.10.21 (the venv's interpreter)
export PYTHONHOME=$(dirname $(dirname $PY)) PYTHONPATH=$V/lib/python3.10/site-packages PYTHONUNBUFFERED=1
mkdir -p $HOME/rest_wl54/tmp; export TMPDIR=$HOME/rest_wl54/tmp
exec $R/ld-linux-x86-64.so.2 --inhibit-cache --library-path $R/lib $PY "$@"
