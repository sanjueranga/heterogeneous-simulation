#!/usr/bin/env bash
# Memory wall with synthetic weights: sweeps worker concurrency inside a container
# with a hard memory cap, and records the peak from the cgroup (memory.peak) and
# whether the kernel OOM-killed it. Faster to run than real_whisper/ since there
# is no torch to install.
#
#   ./run_container.sh
#   LEVELS="1 2 3 4" CAP_GB=5 CPUS=3 ./run_container.sh
#   WEIGHTS_GB=1.0 WORKING_SET_GB=0.4 PREFETCH=1 ./run_container.sh
#
# Take WEIGHTS_GB / WORKING_SET_GB from a docker stats peak on your real worker
# if you want the numbers to mean something.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="memwall-synthetic:latest"
LEVELS="${LEVELS:-1 2 3 4 5 6}"
CAP_GB="${CAP_GB:-5}"
CPUS="${CPUS:-3}"
WEIGHTS_GB="${WEIGHTS_GB:-0.8}"
WORKING_SET_GB="${WORKING_SET_GB:-0.35}"
PREFETCH="${PREFETCH:-1}"
OUT="../data/sim/memory_blowup.csv"

echo "==> Building image ($IMAGE)"
docker build -q -t "$IMAGE" . >/dev/null

echo "==> Sweep -c [$LEVELS] | cap=${CAP_GB}g cpus=${CPUS} | "
echo "    per-worker = ${WEIGHTS_GB}G weights + (1+${PREFETCH}) x ${WORKING_SET_GB}G working set"
echo

printf '%4s | %14s | %15s | %11s | %s\n' "-c" "Peak RSS (GB)" "Per-Worker (GB)" "vs ${CAP_GB}GB cap" "OOM-killed"
printf -- '-%.0s' {1..72}; echo

echo "concurrency,total_rss_gb,per_worker_gb,cap_gb,exceeds_cap,oom_killed,backend" > "$OUT"

for c in $LEVELS; do
  name="figureb_c${c}_$$"
  # memory-swap == memory means no swap, so crossing the cap is a hard OOM
  set +e
  out="$(docker run --name "$name" \
        --memory="${CAP_GB}g" --memory-swap="${CAP_GB}g" --cpus="${CPUS}" \
        "$IMAGE" \
        --concurrency "$c" \
        --weights-gb "$WEIGHTS_GB" \
        --working-set-gb "$WORKING_SET_GB" \
        --prefetch "$PREFETCH" 2>&1)"
  set -e

  oom="$(docker inspect --format '{{.State.OOMKilled}}' "$name" 2>/dev/null || echo "unknown")"
  docker rm -f "$name" >/dev/null 2>&1 || true

  result_line="$(printf '%s\n' "$out" | grep '^RESULT ' || true)"
  if [ -n "$result_line" ]; then
    peak_gb="$(printf '%s' "$result_line" | sed -n 's/.*peak_gb=\([0-9.]*\).*/\1/p')"
    per_worker="$(printf '%s' "$result_line" | sed -n 's/.*per_worker_gb=\([0-9.]*\).*/\1/p')"
  else
    # killed before it printed RESULT, so the peak is the cap
    peak_gb="$CAP_GB.0000"
    per_worker="$(awk -v w="$WEIGHTS_GB" -v ws="$WORKING_SET_GB" -v p="$PREFETCH" \
                  'BEGIN{printf "%.4f", w + (1+p)*ws}')"
    oom="true"
  fi

  exceeds="$(awk -v pk="$peak_gb" -v cap="$CAP_GB" 'BEGIN{print (pk+0 > cap+0) ? "True":"False"}')"
  [ "$oom" = "true" ] && exceeds="True"
  verdict="OK"; [ "$exceeds" = "True" ] && verdict="OOM"

  printf '%4s | %14s | %15s | %11s | %s\n' "$c" "$peak_gb" "$per_worker" "$verdict" "$oom"
  echo "${c},${peak_gb},${per_worker},${CAP_GB}.0,${exceeds},${oom},container-cgroup" >> "$OUT"
done

echo
echo "==> Wrote $OUT"
echo "per-worker footprint (weights + in-flight + prefetched) grows ~linearly with -c,"
echo "and the cgroup cap is a hard OOM kill."
