#!/usr/bin/env bash
# WL104K-C execution recovery (2026-11-24): the first main-chain pass skipped the predeclared N4 reference because its
# "already attempted" test matched the manifest's "order" list. Waits for that pass to end, preserves its decision.json,
# then runs chain_n4fix.sh (chain.sh with only the attempted-test fixed; resumable): only the missing new-scene reference(s),
# their scene replay / analysis and the decision are executed; every completed step is skipped.
# Usage: bash chain_recover_n4.sh <run_root> <first-pass chain.sh MSYS pid>
while kill -0 "$2" 2>/dev/null; do sleep 60; done
tail -1 "$1/status.txt" | grep -qE "^[0-9-]{10} [0-9:]{8} FAILED" && exit 1
[ -f "$1/decision.json" ] && cp "$1/decision.json" "$1/decision_first_pass_before_N4_recovery.json"
echo "$(date '+%F %T') RECOVERY: rerun chain for the skipped predeclared N4 reference (first-pass decision preserved)" >> "$1/status.txt"
bash "$(dirname "$0")/chain_n4fix.sh" "$1"
