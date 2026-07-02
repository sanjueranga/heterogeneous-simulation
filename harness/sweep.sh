#!/usr/bin/env bash
# Runs experiments A, C and B, aggregates into ../data.
# Needs the stack up (docker compose up -d --build), ANTHROPIC_API_KEY in .env
# and a few .mp4 files in videos_in/. Run it from harness/.
set -euo pipefail
cd "$(dirname "$0")"
REPEATS=${REPEATS:-3}
N_SCENES=${N_SCENES:-200}
dcx() { docker compose exec -T "$@"; }

echo "== A: latency crossover =="
for rep in $(seq 1 "$REPEATS"); do
  for K in 1 2 5 10 20; do
    for MODE in v0 v1; do
      dcx web python manage.py submit_batch --k "$K" --mode "$MODE" \
          --run-id "A_${MODE}_k${K}_rep${rep}"
    done
  done
done
python3 probes/aggregate_crossover.py

echo "== C: cache / cost =="
# fragmentation curve one call at a time, then a 4-thread point, plus the caching off/on pair
dcx web python manage.py cost_sweep --scopes 1 2 4 8 16 --n "$N_SCENES" --rate 3 --threads 1 --headline
dcx web python manage.py cost_sweep --scopes 1 4 16 --n "$N_SCENES" --rate 6 --threads 4
cp _data/results/cache_decay.csv ../data/cache_decay.csv
cp _data/results/cache_headline.csv ../data/cache_headline.csv
python3 probes/fit_cache.py

echo "== B: memory (runs on the host) =="
python3 probes/mem_probe.py --heavy-c 1 2 3 4 5 6 --cut-threads 1 2 4 --caps 5 3 --k 6 --repeats "$REPEATS"
python3 probes/aggregate_memory.py --cut-threads 2 --cap 5
