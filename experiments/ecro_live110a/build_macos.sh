#!/usr/bin/env bash
# Build "ReST-GI ECRO Live.app" on macOS (Apple Silicon). NOT yet run on a Mac (WL110A): written and checked on Linux only.
#   bash build_macos.sh --checkpoints <dir containing configs.json, sss/model.pt, near/model.pt>
# Prerequisites: Xcode command line tools, Homebrew `llvm` (Dr.Jit LLVM backend), a Python 3.10-3.12 with Tk
# (python.org installer, or Homebrew python@3.11 + python-tk@3.11).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CK=""
while [ $# -gt 0 ]; do case "$1" in --checkpoints) CK="$2"; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
[ "$(uname -s)" = Darwin ] || { echo "build_macos.sh: macOS only"; exit 1; }
[ -n "$CK" ] || { echo "--checkpoints <dir> is required (the WL100 checkpoints are not in git)"; exit 2; }
PYTHON=${PYTHON:-python3.11}
LLVM_LIB=${ECRO_LIBLLVM:-$(brew --prefix llvm 2>/dev/null)/lib/libLLVM.dylib}
[ -f "$LLVM_LIB" ] || { echo "libLLVM.dylib not found ($LLVM_LIB): brew install llvm"; exit 1; }
"$PYTHON" -c "import tkinter; tkinter.Tk().destroy()" || { echo "$PYTHON has no working Tk"; exit 1; }

B="$HERE/build"; mkdir -p "$B"
[ -d "$B/venv" ] || "$PYTHON" -m venv "$B/venv"
source "$B/venv/bin/activate"
pip install -U pip
pip install -e "$HERE[build]"

rm -rf "$B/runtime"
python "$HERE/tools/stage_runtime.py" "$B/runtime" --repo "$REPO" --checkpoints "$CK"

# gate: the LLVM backend must pass the cross-backend self-check on this Mac before anything is packaged
export REST_ECRO_RUNTIME="$B/runtime" DRJIT_LIBLLVM_PATH="$LLVM_LIB"
python -m rest_ecro_live selfcheck --backend llvm --out "$B/selfcheck_macos.json" | tee "$B/selfcheck_macos.log"
grep -q '^SELFCHECK PASS' "$B/selfcheck_macos.log" || { echo "self-check FAILED on this Mac: not packaging"; exit 1; }

ECRO_RUNTIME="$B/runtime" ECRO_LIBLLVM="$LLVM_LIB" pyinstaller "$HERE/ecro_live_macos.spec" --noconfirm \
    --distpath "$HERE/dist" --workpath "$B/pyi"
APP="$HERE/dist/ReST-GI ECRO Live.app"
unset REST_ECRO_RUNTIME DRJIT_LIBLLVM_PATH
"$APP/Contents/MacOS/rest-ecro-live" info --backend llvm | tee "$B/app_info.json"
echo "built: $APP"
