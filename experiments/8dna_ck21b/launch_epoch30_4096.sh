#!/usr/bin/env bash
set -euo pipefail
BASE=$HOME/rest_8dna_ck21b
PUBLIC=$HOME/rest_8dna_public21a
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
export CK21B_BASE=$BASE CK21B_SOURCE=$HOME/rest_x93d_ck2/code
cd "$BASE"
CHUNK=${1:-8}
RESERVE=${2:-768}
TILES="$BASE/results/epoch30_4096spp_tiles_c${CHUNK}"
exec 9>epoch30_4096.lock
flock -n 9 || exit 75
finish() { echo "$? $(date -Is)" > logs/epoch30_4096.status; }
trap finish EXIT
mkdir -p results/8dna_ck2_epoch30_4096spp
python - <<'PY'
import hashlib,shutil,torch
from pathlib import Path
base=Path.home()/'rest_8dna_ck21b'
source=base/'training_recovery19/CK2_bx38/final.ckpt'
anchor=base/'results/8dna_ck2_epoch30_4096spp/checkpoint.ckpt'
if not anchor.exists(): shutil.copy2(source,anchor)
assert hashlib.sha256(anchor.read_bytes()).hexdigest()=='d107c982ce9546774acf2c0d575c0f151c482a50af1f6f6a1b92e15c7fafd077'
saved=torch.load(anchor,map_location='cpu')
assert saved['global_step']==245760
PY
for target in 64 4096; do
  echo "rendering_${target}spp $(date -Is)" > epoch30_4096.status
  python -u "$PUBLIC/scripts/render.py" --upstream "$UP" \
    --out "$TILES" --asset ck2 --method neural --rep 0 \
    --checkpoint "$BASE/results/8dna_ck2_epoch30_4096spp/checkpoint.ckpt" \
    --scene-bridge "$BASE/scripts/bridge.py" --res 512 --spp "$target" --chunk "$CHUNK" \
    --tile-size 128 --workers auto --gpu-reserve-mib "$RESERVE" --task-timeout 120
done
echo "figure_export $(date -Is)" > epoch30_4096.status
python -u scripts/finalize_epoch30_highspp.py "$TILES"
echo "complete $(date -Is)" > epoch30_4096.status
