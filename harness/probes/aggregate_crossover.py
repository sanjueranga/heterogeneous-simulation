#!/usr/bin/env python3
"""batch_runs.csv -> ../data/crossover.csv (K, v0_serial_s, v1_fleet_s, speedup),
using the median wall time over repeats for each (mode, K).

  python3 probes/aggregate_crossover.py
"""
import csv
import pathlib
import statistics
from collections import defaultdict

HARNESS = pathlib.Path(__file__).resolve().parent.parent
SRC = HARNESS / "_data" / "results" / "batch_runs.csv"
OUT = HARNESS.parent / "data" / "crossover.csv"


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} — run Experiment A first")
    walls = defaultdict(list)   # (mode, K) -> [wall_s]
    with open(SRC) as f:
        for r in csv.DictReader(f):
            if r.get("run_id", "").startswith("warm"):  # skip warm-up runs
                continue
            if int(r["completed"]) >= int(r["k"]):     # only batches that finished
                walls[(r["mode"], int(r["k"]))].append(float(r["wall_s"]))

    ks = sorted({k for (_, k) in walls})
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "v0_serial_s", "v1_fleet_s", "speedup"])
        for k in ks:
            v0 = walls.get(("v0", k))
            v1 = walls.get(("v1", k))
            if not v0 or not v1:
                continue
            s0 = statistics.median(v0)
            s1 = statistics.median(v1)
            w.writerow([k, round(s0, 4), round(s1, 4), round(s0 / s1, 4)])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
