#!/usr/bin/env bash
set -euo pipefail
R=$HOME/rest_wl56
while [[ "$(cat "$R/launcher/reference.status")" == running ]]; do sleep 10; done
[[ "$(cat "$R/launcher/reference.status")" == done ]]
exec bash "$R/repo/experiments/fnto_sel56/launch_primary.sh"
