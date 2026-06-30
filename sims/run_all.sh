#!/usr/bin/env bash
# Runs all the simulations and writes their CSVs to ../data/sim.
#   ./run_all.sh
#   MEM_LEVELS=1,2 FOOTPRINT_GB=1.0 ./run_all.sh    # lighter memory run
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
MEM_LEVELS="${MEM_LEVELS:-1,2,3,4}"
FOOTPRINT_GB="${FOOTPRINT_GB:-2.0}"

"$PY" -m pip install -q -r requirements.txt

"$PY" model_analytical.py
"$PY" measure_crossover.py
"$PY" measure_cache_decay.py
"$PY" measure_autoscaling.py
"$PY" measure_weighted_scaling.py
# this one really allocates RAM (about FOOTPRINT_GB per worker)
"$PY" measure_memory_blowup.py --levels "$MEM_LEVELS" --max-array-gb "$FOOTPRINT_GB"

echo
echo "done, csvs are in ../data/sim"
