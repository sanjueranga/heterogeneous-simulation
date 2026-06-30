#!/usr/bin/env python3
"""
Sim for prompt-cache hit rate vs number of cache scopes.

Every classification call starts with the same long prefix (system prompt plus
the full taxonomy), marked as cacheable. A call hits the cache if an earlier call
touched the same warm prefix within the cache TTL. If the same request stream is
spread over C independent scopes, each scope sees only 1/C of the traffic, so
the prefix goes cold between calls and the hit rate drops, even though the number
of calls is the same.

Model: each scope is a Poisson stream, rate lambda/C, TTL tau. A call hits iff
the gap since the previous call on its scope is < tau. The hit rate converges to

    p_hit(C) = 1 - exp(-rho / C),    rho = lambda * tau

(derivation in MODEL.md). rho is the only knob. Fit it from measured data with
--calibrate (csv with columns instances,p_hit): least squares on
-ln(1 - p_hit) = rho / C through the origin. No API calls are made here; the
real measurement is `manage.py cost_sweep` in harness/.

    python3 measure_cache_decay.py
    python3 measure_cache_decay.py --rho 6 --trials 200000
    python3 measure_cache_decay.py --calibrate ../data/cache_decay.csv
"""

import argparse
import csv
import math
import os
import random

# rho = lambda * tau = requests per cache lifetime. Illustrative default, fit it
# with --calibrate if you have real numbers.
RHO_DEFAULT = 6.0

# $ per call, roughly list-price shaped (cached read ~10x cheaper than full input)
CALLS_PER_BATCH = 100
C_CACHED = 0.0004
C_FULL = 0.0030

# ms per call
L_CACHED = 120.0
L_FULL = 850.0

FLEET_SIZES = [1, 2, 4, 8, 16]
DEFAULT_TRIALS = 200000  # Monte-Carlo arrivals per scope; >=1e5 gives ~3 d.p.
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim", "cache_decay.csv")


def p_hit_montecarlo(c, rho, trials, rng):
    """Empirical hit rate for C scopes. tau = 1, so lambda = rho and the gap
    between calls on one scope is Exponential(mean C/rho)."""
    mean_gap = c / rho
    hits = 0
    for _ in range(trials):
        gap = rng.expovariate(1.0 / mean_gap)
        if gap < 1.0:                        # previous arrival within TTL tau=1
            hits += 1
    return hits / trials


def p_hit_closed_form(c, rho):
    return 1.0 - math.exp(-rho / c)


def fit_rho_from_csv(path):
    """Fit rho on -ln(1 - p_hit) = rho/C. Rows with p_hit >= 1 are skipped."""
    import numpy as np
    xs, ys = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            c = float(row["instances"])
            p = float(row["p_hit"])
            if 0.0 < p < 1.0:
                xs.append(1.0 / c)
                ys.append(-math.log(1.0 - p))
    if len(xs) < 2:
        raise ValueError("Need >=2 calibration points with 0 < p_hit < 1.")
    x = np.array(xs)
    y = np.array(ys)
    rho = float((x @ y) / (x @ x))           # slope through origin
    return rho, len(xs)


def cost_per_batch(ph):
    return CALLS_PER_BATCH * (ph * C_CACHED + (1.0 - ph) * C_FULL)


def latency_ms(ph):
    return ph * L_CACHED + (1.0 - ph) * L_FULL


def main():
    parser = argparse.ArgumentParser(description="Prompt-cache hit rate vs cache scopes.")
    parser.add_argument("--calibrate", metavar="CSV", default=None,
                        help="CSV with columns instances,p_hit to fit rho from real data.")
    parser.add_argument("--rho", type=float, default=RHO_DEFAULT,
                        help="lambda*tau (requests per cache lifetime). Default illustrative.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="Monte-Carlo arrivals per scope.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.calibrate:
        rho, n = fit_rho_from_csv(args.calibrate)
        source = f"calibrated (rho={rho:.4f} from {n} measured points in {args.calibrate})"
    else:
        rho = args.rho
        source = f"default rho={rho:.4f} (not fitted, see --calibrate)"

    rng = random.Random(args.seed)

    print("\nPrompt-cache hit rate vs scopes (Monte-Carlo of p_hit = 1 - e^(-rho/C))")
    print(f"rho: {source}\n")
    header = (f"{'Scopes C':>8} | {'p_hit (sim)':>11} | {'p_hit (law)':>11} | "
              f"{'Latency (ms)':>12} | {'Cost ($/batch)':>14}")
    print(header)
    print("-" * len(header))

    rows = []
    for c in FLEET_SIZES:
        ph = p_hit_montecarlo(c, rho, args.trials, rng)
        ph_law = p_hit_closed_form(c, rho)
        lat = latency_ms(ph)
        cost = cost_per_batch(ph)
        print(f"{c:>8} | {ph:>11.3f} | {ph_law:>11.3f} | {lat:>12.1f} | {cost:>14.4f}")
        rows.append({
            "instances": c,
            "p_hit": round(ph, 4),
            "p_hit_law": round(ph_law, 4),
            "avg_latency_ms": round(lat, 2),
            "avg_cost_per_batch": round(cost, 6),
        })

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "instances", "p_hit", "p_hit_law", "avg_latency_ms", "avg_cost_per_batch"])
        writer.writeheader()
        writer.writerows(rows)

    base, worst = rows[0], rows[-1]
    print(f"\nFrom C=1 to C={worst['instances']}: "
          f"hit rate {base['p_hit']:.3f} -> {worst['p_hit']:.3f}, "
          f"cost x{worst['avg_cost_per_batch']/base['avg_cost_per_batch']:.2f}, "
          f"latency x{worst['avg_latency_ms']/base['avg_latency_ms']:.2f}.")
    print("Keeping everything on one warm scope (C=1) keeps cost flat however much "
          "you parallelise the calls.")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
