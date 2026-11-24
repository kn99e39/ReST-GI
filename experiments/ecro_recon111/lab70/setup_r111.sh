#!/usr/bin/env bash
# WL111 host setup on LabServer70 (no GPU, trivial CPU): output tree and read-only links to the frozen inputs already here.
set -eu
R=$HOME/rest_sss101a
O=$R/results/ecro_recon111
mkdir -p "$O/inputs" "$O/mpl" "$O/tmp" "$O/triton"
cd "$O/inputs"
for x in wl101 results evaluation; do [[ -e $x ]] || ln -s "$R/results/ecro_denoise108/inputs/$x" "$x"; done
[[ -e capture ]] || ln -s "$R/results/ecro_denoise107/capture" capture
mkdir -p wl101b_primary phys110
ls -l "$O/inputs"
