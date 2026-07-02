#!/usr/bin/env python3
"""memory_blowup_detail.csv -> ../data/memory_blowup.csv, plus a note on which
configs were safe.

The figure needs one row per concurrency, so this takes one slice (--cut-threads,
--cap) and the median peak over repeats. It also scans the whole sweep for the
largest (heavy_c, cut_threads, cap) that never OOM-killed or went over the cap.

  python3 probes/aggregate_memory.py --cut-threads 2 --cap 5
"""
import argparse
import csv
import pathlib
import statistics
from collections import defaultdict

HARNESS = pathlib.Path(__file__).resolve().parent.parent
SRC = HARNESS / "_data" / "results" / "memory_blowup_detail.csv"
OUT = HARNESS.parent / "data" / "memory_blowup.csv"
CLEAN = ["concurrency", "total_rss_gb", "per_worker_gb", "cap_gb",
         "exceeds_cap", "oom_killed", "backend"]


def _b(s):
    return str(s).strip().lower() in ("true", "1", "yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut-threads", type=int, default=2)
    ap.add_argument("--cap", type=float, default=5)
    o = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} — run mem_probe.py first")

    rows = list(csv.DictReader(open(SRC)))

    # one slice -> memory_blowup.csv
    slice_rows = defaultdict(list)  # concurrency -> [row]
    for r in rows:
        if int(r["cut_threads"]) == o.cut_threads and float(r["cap_gb"]) == o.cap:
            slice_rows[int(r["concurrency"])].append(r)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLEAN)
        w.writeheader()
        for c in sorted(slice_rows):
            rs = slice_rows[c]
            rss = statistics.median(float(r["total_rss_gb"]) for r in rs)
            w.writerow({
                "concurrency": c, "total_rss_gb": round(rss, 4),
                "per_worker_gb": round(rss / max(1, c), 4), "cap_gb": o.cap,
                "exceeds_cap": rss > o.cap,
                "oom_killed": "true" if any(_b(r["oom_killed"]) for r in rs) else "false",
                "backend": "real-whisper",
            })
    print(f"wrote {OUT} (cut_threads={o.cut_threads}, cap={o.cap}g)")

    # safe configs across the whole sweep
    ok = defaultdict(lambda: True)   # (heavy_c, cut_threads, cap) -> safe in all reps
    for r in rows:
        key = (int(r["concurrency"]), int(r["cut_threads"]), float(r["cap_gb"]))
        if _b(r["oom_killed"]) or _b(r["exceeds_cap"]):
            ok[key] = False
        else:
            ok[key] = ok[key]
    safe = [k for k, v in ok.items() if v]
    print("\nOOM-SAFE configs (heavy_c, cut_threads, cap_gb) — no OOM/over-cap in any rep:")
    for cap in sorted({k[2] for k in safe}):
        best = max([k for k in safe if k[2] == cap], key=lambda k: k[0], default=None)
        if best:
            print(f"  cap {cap:.0f}g -> max safe heavy_c = {best[0]} (cut_threads={best[1]})")
    if safe:
        overall = max(safe, key=lambda k: (k[0], k[1]))
        print(f"\n  RECOMMENDED OOM-safe setup: heavy_c={overall[0]}, "
              f"cut_threads={overall[1]}, cap={overall[2]:.0f}g")


if __name__ == "__main__":
    main()
