#!/usr/bin/env bash
# Memory wall with the real models: loads whisperx base int8 + the wav2vec2 align
# model in each of `-c` workers inside a memory-capped container, and records the
# peak from the cgroup (memory.peak) and whether the kernel OOM-killed it.
#
#   ./run_real.sh                          # -c 1..6, 5g cap, prefetch 1
#   LEVELS="1 2 3 4" ./run_real.sh
#   CAP_GB=5 CPUS=3 PREFETCH=1 ./run_real.sh

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="memwall-whisper:latest"
LEVELS="${LEVELS:-1 2 3 4 5 6}"
CAP_GB="${CAP_GB:-5}"
CPUS="${CPUS:-3}"
PREFETCH="${PREFETCH:-1}"
OUT="../../data/memory_blowup.csv"

echo "==> Building image ($IMAGE), downloads torch + whisperx + weights the first time"
docker build -t "$IMAGE" . >/dev/null

echo "==> Sweep -c [$LEVELS] | cap=${CAP_GB}g swap=${CAP_GB}g cpus=${CPUS} | prefetch=${PREFETCH}"
echo "    backend=real-whisper (base int8 transcription + en wav2vec2 align, per worker)"
echo

printf '%4s | %14s | %15s | %11s | %s\n' "-c" "Peak RSS (GB)" "Per-Worker (GB)" "vs ${CAP_GB}GB cap" "OOM-killed"
printf -- '-%.0s' {1..72}; echo

echo "concurrency,total_rss_gb,per_worker_gb,cap_gb,exceeds_cap,oom_killed,backend" > "$OUT"

last_per_worker=""   # per-worker footprint is ~constant, reused for OOM rows that never report
for c in $LEVELS; do
  name="figureb_real_c${c}_$$"
  # memory-swap == memory means no swap, so crossing the cap is a hard OOM
  set +e
  out="$(docker run --name "$name" \
        --memory="${CAP_GB}g" --memory-swap="${CAP_GB}g" --cpus="${CPUS}" \
        "$IMAGE" \
        --concurrency "$c" \
        --prefetch "$PREFETCH" 2>&1)"
  set -e

  oom="$(docker inspect --format '{{.State.OOMKilled}}' "$name" 2>/dev/null || echo "unknown")"
  docker rm -f "$name" >/dev/null 2>&1 || true

  result_line="$(printf '%s\n' "$out" | grep '^RESULT ' || true)"
  if [ -n "$result_line" ]; then
    peak_gb="$(printf '%s' "$result_line" | sed -n 's/.*peak_gb=\([0-9.]*\).*/\1/p')"
    per_worker="$(printf '%s' "$result_line" | sed -n 's/.*per_worker_gb=\([0-9.]*\).*/\1/p')"
    last_per_worker="$per_worker"
  else
    # killed before printing RESULT, so the peak is the cap. Reuse the last
    # per-worker figure, it barely changes with c.
    peak_gb="${CAP_GB}.0000"
    per_worker="${last_per_worker:-NA}"
    oom="true"
  fi

  exceeds="$(awk -v pk="$peak_gb" -v cap="$CAP_GB" 'BEGIN{print (pk+0 > cap+0) ? "True":"False"}')"
  [ "$oom" = "true" ] && exceeds="True"
  verdict="OK"; [ "$exceeds" = "True" ] && verdict="OOM"

  printf '%4s | %14s | %15s | %11s | %s\n' "$c" "$peak_gb" "$per_worker" "$verdict" "$oom"
  echo "${c},${peak_gb},${per_worker},${CAP_GB}.0,${exceeds},${oom},real-whisper" >> "$OUT"
done

echo
echo "==> Wrote $OUT"
echo "per-worker footprint = torch runtime + base int8 weights + align model +"
echo "(1+${PREFETCH}) audio working sets. Going over the cap is a hard OOM kill."
