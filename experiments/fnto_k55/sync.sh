#!/usr/bin/env bash
# Copy code to LabServer69 (~/rest_wl55/repo). phys_ref53 and fnto_c54 come from `git archive HEAD` (committed state only; uncommitted edits by
# other sessions are not shipped); fnto_k55 comes from the working tree and the dirty state is recorded.
#   experiments/fnto_k55/sync.sh push            code -> server
#   experiments/fnto_k55/sync.sh pull <sub>      ~/rest_wl55/out/<sub> -> results/fnto_k55/<sub> (json, txt, png, npz < 50 MB)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); H=LabServer69
case "$1" in
 push)
  ssh $H 'mkdir -p ~/rest_wl55/repo ~/rest_wl55/out ~/rest_wl55/launcher ~/rest_wl55/tmp'
  git -C "$ROOT" archive HEAD experiments/phys_ref53 experiments/fnto_c54 | ssh $H 'tar -C ~/rest_wl55/repo -xf -'
  tar -C "$ROOT" --exclude=__pycache__ -cf - experiments/fnto_k55 | ssh $H 'tar -C ~/rest_wl55/repo -xf -'
  echo "$(git -C "$ROOT" rev-parse HEAD) fnto_k55_dirty=$(git -C "$ROOT" status --porcelain experiments/fnto_k55 | wc -l | tr -d ' ')" | ssh $H 'cat > ~/rest_wl55/repo_commit.txt' ;;
 pull)
  mkdir -p "$ROOT/results/fnto_k55/$2"
  ssh $H "cd ~/rest_wl55/out/$2 && find . -type f -size -50000k ! -name '*.pt' | tar -cf - -T -" | tar -C "$ROOT/results/fnto_k55/$2" -xf - ;;
esac
