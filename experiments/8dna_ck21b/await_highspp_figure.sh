#!/usr/bin/env bash
set -euo pipefail
BASE=$HOME/rest_8dna_ck21b
while [ ! -f "$BASE/results/8dna_ck2_epoch18_4096spp/complete.json" ]; do sleep 10; done
exec "$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python" -u "$BASE/scripts/figure_highspp_matched.py"
