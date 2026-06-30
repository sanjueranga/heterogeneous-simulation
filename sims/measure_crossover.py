#!/usr/bin/env python3
"""
Sim for the latency crossover: serial monolith (V0) vs fork-join fleet (V1).

Stage times are scaled way down (fractions of a second) but keep roughly the
same ratios as the real pipeline, and the crossover depends on the ratios, not
the absolutes. The real-pipeline version of this is harness/ (experiment A).

V0: one task at a time, whisper model reloaded for every video,
    transcribe -> cut -> classify run back to back.
V1: heavy pool (-c 4) transcribes, forks classify to the light pool (-c 10) and
    cuts meanwhile. Model load is paid once per worker child instead of per task,
    and there's a one-off broker/worker warmup cost.

So V0 wins at K=1 (no warmup to pay off) and loses as K grows.

    python3 measure_crossover.py
"""

import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# stage times in seconds (scaled down)
MODEL_LOAD   = 0.30   # whisper load, paid per task in V0
T_TRANSCRIBE = 0.10
T_CUT        = 0.18   # ffmpeg, cpu heavy
T_CLASSIFY   = 0.12   # network / LLM call

# V1 fleet shape
HEAVY_C = 4
LIGHT_C = 10
MAX_TASKS_PER_CHILD = 10   # worker child gets recycled (and reloads the model) after this many tasks
BROKER_WARMUP = 0.40       # broker connect + cold worker spin-up

BATCH_SIZES = [1, 2, 5, 10, 20]
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim", "crossover.csv")


def run_v0_serial(k):
    start = time.perf_counter()
    for _ in range(k):
        time.sleep(MODEL_LOAD)     # reloaded every video
        time.sleep(T_TRANSCRIBE)
        time.sleep(T_CUT)
        time.sleep(T_CLASSIFY)
    return time.perf_counter() - start


_tls = threading.local()


def _heavy_video(light_pool):
    # model load once per worker child, again only when the child is recycled
    n = getattr(_tls, "tasks_done", 0)
    if n % MAX_TASKS_PER_CHILD == 0:
        time.sleep(MODEL_LOAD)
    _tls.tasks_done = n + 1

    time.sleep(T_TRANSCRIBE)
    classify_future = light_pool.submit(time.sleep, T_CLASSIFY)   # fork to light pool
    time.sleep(T_CUT)                                             # cut keeps going on heavy
    classify_future.result()                                      # join


def run_v1_fork_join(k):
    start = time.perf_counter()
    time.sleep(BROKER_WARMUP)  
    with ThreadPoolExecutor(max_workers=HEAVY_C) as heavy, \
         ThreadPoolExecutor(max_workers=LIGHT_C) as light:
        futures = [heavy.submit(_heavy_video, light) for _ in range(k)]
        for f in futures:
            f.result()
    return time.perf_counter() - start


def main():
    rows = []
    crossover_k = None

    print("\nV0 serial monolith vs V1 fork-join fleet (simulated)\n")
    header = f"{'K':>4} | {'V0 serial (s)':>13} | {'V1 fleet (s)':>12} | {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for k in BATCH_SIZES:
        v0 = run_v0_serial(k)
        v1 = run_v1_fork_join(k)
        speedup = v0 / v1 if v1 > 0 else float("inf")
        if crossover_k is None and v1 < v0:
            crossover_k = k
        print(f"{k:>4} | {v0:>13.3f} | {v1:>12.3f} | {speedup:>7.2f}x")
        rows.append({
            "K": k,
            "v0_serial_s": round(v0, 4),
            "v1_fleet_s": round(v1, 4),
            "speedup": round(speedup, 4),
        })

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["K", "v0_serial_s", "v1_fleet_s", "speedup"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    if crossover_k is not None:
        print(f"Crossover: V1 fleet overtakes V0 monolith at K = {crossover_k}.")
        print("V0 runs one video at a time and reloads the model each time; V1 "
              "overlaps cut/classify and loads the model once per worker child.")
    else:
        print("Crossover: V1 did not overtake V0 in the tested range.")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
