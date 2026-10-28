#!/usr/bin/env bash
# Copy the WL54 code tree to LabServer69 (tar over ssh; no git on the server copy) and record the commit.
#   experiments/fnto_c54/sync.sh push      local worktree -> ~/rest_wl54/repo
#   experiments/fnto_c54/sync.sh pull      ~/rest_wl54/out/<sub> -> results/fnto_c54/<sub>   (small files only: json, txt, png, npz < 50 MB)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); H=LabServer69
case "$1" in
 push)
  git -C "$ROOT" rev-parse HEAD > /tmp/wl54_commit.txt
  ssh $H 'mkdir -p ~/rest_wl54/repo ~/rest_wl54/out ~/rest_wl54/launcher'
  tar -C "$ROOT" --exclude=__pycache__ -cf - experiments/phys_ref53 experiments/fnto_c54 | ssh $H 'tar -C ~/rest_wl54/repo -xf -'
  scp -q /tmp/wl54_commit.txt $H:rest_wl54/repo_commit.txt ;;
 pull)
  mkdir -p "$ROOT/results/fnto_c54/$2"
  ssh $H "cd ~/rest_wl54/out/$2 && find . -type f -size -50000k ! -name '*.pt' | tar -cf - -T -" | tar -C "$ROOT/results/fnto_c54/$2" -xf - ;;
esac
