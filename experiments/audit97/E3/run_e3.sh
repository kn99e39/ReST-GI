#!/bin/bash
# WL97 audit track E3. usage on LabServer63: run_e3.sh <script.py> [args...]   (own directory ~/rest_x97/E3 only)
. ~/rest_x95a/env.sh
export PYTHONDONTWRITEBYTECODE=1
cd ~/rest_x97/E3
exec $PY "$@"
