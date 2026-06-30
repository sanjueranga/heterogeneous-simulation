#!/usr/bin/env python3
"""
Sim for the memory wall: N prefork workers, each with its own copy of the model
weights, so total RSS grows roughly linearly with concurrency and eventually
passes the container's memory cap.

Each worker tries to load faster-whisper "base" (int8). If that isn't installed
it allocates and touches a ~2 GB numpy array as a stand-in. Every worker reports
its own RSS once loaded, and the total is summed with the parent's.

Heads up: with the default 2 GB footprint, -c 4 needs ~8 GB. To smoke test:

    python3 measure_memory_blowup.py --levels 1,2
    python3 measure_memory_blowup.py --max-array-gb 1.0

The real version of this (actual whisperx workers in a capped container) lives
in container/real_whisper.
"""

import argparse
import csv
import multiprocessing as mp
import os
import time

import psutil

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim", "memory_blowup.csv")
GiB = 1024 ** 3

HEAVY_CONTAINER_CAP_GB = 5.0
CONFIGURED_HEAVY_C = 4   # the concurrency the heavy worker is actually run at
# Stand-in for a bigger in-process model. faster-whisper base int8 on its own is
# only ~0.2-0.3 GB, so the wall shows up with larger models. Use docker stats to
# get a real number for your setup and pass it as --max-array-gb.
DEFAULT_FOOTPRINT_GB = 2.0


def _worker(hold_seconds, max_array_gb, result_q):
    """Load weights, measure THIS process's own RSS, report it, then idle."""
    backend = "numpy-fallback"
    keep_alive = None
    try:
        from faster_whisper import WhisperModel  # noqa: WPS433
        keep_alive = WhisperModel("base", device="cpu", compute_type="int8")
        backend = "faster-whisper:base"
    except Exception:
        import numpy as np
        n = int((max_array_gb * GiB) / 8)
        # random data on purpose: macOS/zram compress uniform pages, so a
        # constant-filled array would never show up in RSS
        keep_alive = np.random.default_rng().random(n)

    # each worker reports its own RSS; walking children() from the parent races startup
    rss = psutil.Process().memory_info().rss
    result_q.put((backend, rss))
    time.sleep(hold_seconds)
    del keep_alive


def measure_level(m, max_array_gb, hold_seconds=4.0):
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(hold_seconds, max_array_gb, result_q))
        for _ in range(m)
    ]
    for p in procs:
        p.start()

    backends, worker_rss = set(), 0
    for _ in range(m):
        backend, rss = result_q.get(timeout=120)  # blocks until each worker loads
        backends.add(backend)
        worker_rss += rss

    total = psutil.Process().memory_info().rss + worker_rss

    for p in procs:
        p.join(timeout=hold_seconds + 5)
        if p.is_alive():
            p.terminate()
            p.join()

    return total, (", ".join(sorted(backends)) if backends else "unknown")


def main():
    parser = argparse.ArgumentParser(description="Total RSS vs heavy-worker concurrency.")
    parser.add_argument("--levels", default="1,2,3,4",
                        help="Comma-separated prefork concurrencies (default: 1,2,3,4).")
    parser.add_argument("--max-array-gb", type=float, default=DEFAULT_FOOTPRINT_GB,
                        help=f"Per-worker fallback footprint in GB (default: {DEFAULT_FOOTPRINT_GB}).")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    print("\nTotal RSS vs heavy-worker concurrency (synthetic weights)")
    print(f"container cap: {HEAVY_CONTAINER_CAP_GB:.1f} GB  |  "
          f"configured: -c {CONFIGURED_HEAVY_C}  |  "
          f"per-worker footprint: ~{args.max_array_gb:.1f} GB\n")

    header = (f"{'-c':>4} | {'Total RSS (GB)':>14} | {'Per-Worker (GB)':>15} | "
              f"{'vs 5GB cap':>11} | {'Backend':<22}")
    print(header)
    print("-" * len(header))

    rows = []
    for m in levels:
        total_bytes, backend = measure_level(m, args.max_array_gb)
        total_gb = total_bytes / GiB
        per_proc = total_gb / m if m else 0.0
        verdict = "OOM" if total_gb > HEAVY_CONTAINER_CAP_GB else "ok"
        print(f"{m:>4} | {total_gb:>14.2f} | {per_proc:>15.2f} | "
              f"{verdict:>11} | {backend:<22}")
        rows.append({
            "concurrency": m,
            "total_rss_gb": round(total_gb, 4),
            "per_worker_gb": round(per_proc, 4),
            "cap_gb": HEAVY_CONTAINER_CAP_GB,
            "exceeds_cap": total_gb > HEAVY_CONTAINER_CAP_GB,
            "backend": backend,
        })

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "concurrency", "total_rss_gb", "per_worker_gb",
            "cap_gb", "exceeds_cap", "backend"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {CSV_PATH}")
    print("Per-worker weights, no sharing -> RSS grows ~linearly with -c. "
          f"Past {HEAVY_CONTAINER_CAP_GB:.0f} GB the container gets OOM-killed.")


if __name__ == "__main__":
    main()
