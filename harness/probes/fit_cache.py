#!/usr/bin/env python3
"""Fit rho to the measured (instances, p_hit) points and fill the p_hit_law
column of ../data/cache_decay.csv in place. The measured p_hit values are left
alone (measure_cache_decay.py --calibrate would overwrite them with simulated
ones).

  python3 probes/fit_cache.py

Fit: -ln(1 - p_hit) = rho / C, least squares through the origin. Rows with
p_hit <= 0 or >= 1 are skipped.
"""
import csv
import math
import pathlib

HARNESS = pathlib.Path(__file__).resolve().parent.parent
CSV = HARNESS.parent / "data" / "cache_decay.csv"


def main():
    rows = list(csv.DictReader(open(CSV)))
    xs, ys = [], []
    for r in rows:
        c = float(r["instances"])
        p = float(r["p_hit"])
        if 0.0 < p < 1.0 and c > 0:
            xs.append(1.0 / c)
            ys.append(-math.log(1.0 - p))
    if len(xs) < 2:
        raise SystemExit("need >=2 measured points with 0<p_hit<1 to fit rho")
    rho = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)

    for r in rows:
        c = float(r["instances"])
        r["p_hit_law"] = round(1.0 - math.exp(-rho / c), 4) if c > 0 else ""

    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instances", "p_hit", "p_hit_law",
                                          "avg_latency_ms", "avg_cost_per_batch"])
        w.writeheader()
        w.writerows(rows)

    lo = min(rows, key=lambda r: float(r["instances"]))
    hi = max(rows, key=lambda r: float(r["instances"]))
    print(f"fitted rho = {rho:.4f} from {len(xs)} measured points")
    print(f"measured p_hit: C={lo['instances']} -> {lo['p_hit']}  ...  "
          f"C={hi['instances']} -> {hi['p_hit']}")
    print(f"wrote p_hit_law into {CSV} (measured p_hit preserved)")


if __name__ == "__main__":
    main()
