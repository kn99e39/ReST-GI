#!/usr/bin/env bash
# Waits for the Stage A chain to end (DONE or FAILED in status.txt), then runs the main chain. Usage: bash chain_after_A.sh <run_root>
until grep -q "STAGE A DONE\|FAILED" "$1/status.txt"; do sleep 60; done
grep -q "FAILED" "$1/status.txt" && { echo "$(date '+%F %T') main chain not started: Stage A FAILED" >> "$1/status.txt"; exit 1; }
bash "$(dirname "$0")/chain.sh" "$1"
