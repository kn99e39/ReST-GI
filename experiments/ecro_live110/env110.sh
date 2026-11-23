# WL110 live demonstrator environment on LabServer69 (same stack as WL101 arm B, Amendment 1). Source this file.
W=${DYN101_WORKSPACE:-$HOME/rest_dyn101}
export LIVE110_PY=${LIVE110_PY:-$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python}
export DYN101_MI_DIR=$W/external/mi_3.8.0 DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH=$W/external/mi39_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
