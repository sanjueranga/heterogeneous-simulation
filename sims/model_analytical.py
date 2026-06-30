#!/usr/bin/env python3
"""
Closed-form latency and cost model for serial vs fork-join.

Derives (1) the latency crossover between the serial monolith (V0) and the
fork-join fleet (V1), and (2) how prompt-cache hit rate, and so cost, falls when
classification is spread over several cache scopes. measure_crossover.py and
measure_cache_decay.py are simulation checks of these equations. Symbols are in
MODEL.md. Constants match measure_crossover.py so the two can be compared.

1.  LATENCY MODEL
Per-clip stage times (seconds):  l (model load), t_t (transcribe), t_u (cut),
t_c (classify).  p = heavy parallelism, B = one-time broker/cold-start warmup.

V0 serial monolith -- one clip at a time, model reloaded per clip, no overlap:

        L_serial(K) = K * (l + t_t + t_u + t_c)                         (1)

V1 fork-join fleet -- p heavy workers; per clip, classify is forked to the light
pool so it OVERLAPS the cut (cut ∥ classify -> max(t_u, t_c)); model load is paid
once per worker (amortized by max_tasks_per_child):

        L_fleet(K) = B + l + ceil(K/p) * ( t_t + max(t_u, t_c) )        (2)

Speedup:        S(K) = L_serial(K) / L_fleet(K)                          (3)

Crossover (drop the ceil for the continuous root): smallest K with L_fleet<L_serial

        K* = (B + l) / [ (l + t_t + t_u + t_c) - (t_t + max(t_u,t_c))/p ]   (4)

Asymptotic speedup (K -> inf):

        S_inf = (l + t_t + t_u + t_c) / [ (t_t + max(t_u,t_c)) / p ]    (5)

S_inf can exceed p: the serial path also re-pays model load every clip and runs
classify in series, both of which the fleet hides.

2.  CACHE-EFFICIENCY (COST) MODEL
Running classification on 4 workers does not cost 4x the tokens, a batch of K
clips is still K calls. Parallelism only raises cost indirectly, by fragmenting
the prompt cache.

Derivation (queuing).  Classification requests share a long common prefix (system
prompt + taxonomy), so a call is "cached" iff a prior call hit the SAME cache scope
within the cache lifetime tau. Total request rate lambda is split over C
independent cache scopes (instances / connections / regions) -> each scope sees a
Poisson stream of rate lambda/C. For a Poisson process, the probability that the
preceding arrival on a scope falls within tau is:

        p_hit(C) = 1 - exp( - lambda*tau / C ) = 1 - exp( -rho / C )    (6)

with rho = lambda*tau  (fleet-wide expected requests per cache lifetime).
  C = 1   -> p_hit = 1 - e^{-rho}      (one warm scope; high if rho >> 1)
  C >> rho-> p_hit ~ rho/C -> 0        (spread too thin; caches go cold)

Per-call cost mixes cached (c_h) and full (c_f > c_h) prices:

        cost_call(C) = c_f - (c_f - c_h) * p_hit(C)                     (7)
        Cost(K, C)   = K * cost_call(C)                                 (8)

Cost penalty of fragmenting from 1 to C scopes (the price of careless parallelism):

        dCost(C) = K (c_f - c_h) [ p_hit(1) - p_hit(C) ]
                 = K (c_f - c_h) [ e^{-rho/C} - e^{-rho} ]             (9)

Per-call latency behaves the same way (cached calls have lower TTFT):

        lat_call(C) = l_f - (l_f - l_h) * p_hit(C)                      (10)

3.  THE DESIGN PRINCIPLE
Latency wants high parallelism p (eq. 2: L ~ 1/p). Cost wants few cache scopes C
(eq. 8). They are separate knobs: p is execution concurrency, C is the number of
cache scopes. If the provider's cache is keyed on content and shared across
workers, C = 1 regardless of p, so you get the latency win without the cost
penalty. Route classification so workers share one warm scope and keep p high.
"""

import argparse
import csv
import math
import os

# latency constants, same as measure_crossover.py
L_LOAD = 0.30        # l   : model load (per clip in V0; amortized in V1)
T_T = 0.10           # t_t : transcribe
T_U = 0.18           # t_u : cut (FFmpeg)
T_C = 0.12           # t_c : classify
P = 4                # p   : heavy parallelism (-c 4)
B_WARMUP = 0.40      # B   : one-time broker + cold-start warmup

# cache / cost constants (illustrative)
RHO = 6.0            # rho = lambda*tau (requests per cache lifetime, fleet-wide)
C_FULL = 0.0030      # c_f : full (uncached) call cost
C_CACHED = 0.0004    # c_h : cached call cost
L_FULL = 850.0       # l_f : full call latency (ms)
L_CACHED = 120.0     # l_h : cached call latency (ms)

BATCH_SIZES = [1, 2, 5, 10, 20]
FLEET_SCOPES = [1, 2, 4, 8, 16]
CSV_LATENCY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim", "model_latency.csv")
CSV_COST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim", "model_cost.csv")


# ---- Latency (eqs 1-5) ----
def l_serial(k):
    return k * (L_LOAD + T_T + T_U + T_C)


def l_fleet(k):
    return B_WARMUP + L_LOAD + math.ceil(k / P) * (T_T + max(T_U, T_C))


def speedup(k):
    return l_serial(k) / l_fleet(k)


def crossover_k():
    denom = (L_LOAD + T_T + T_U + T_C) - (T_T + max(T_U, T_C)) / P
    k_star = (B_WARMUP + L_LOAD) / denom
    return k_star, math.ceil(k_star)


def asymptotic_speedup():
    return (L_LOAD + T_T + T_U + T_C) / ((T_T + max(T_U, T_C)) / P)


# ---- Cache / cost (eqs 6-10) ----
def p_hit(c, rho=RHO):
    return 1.0 - math.exp(-rho / c)


def cost_call(c, rho=RHO):
    return C_FULL - (C_FULL - C_CACHED) * p_hit(c, rho)


def lat_call(c, rho=RHO):
    return L_FULL - (L_FULL - L_CACHED) * p_hit(c, rho)


def main():
    parser = argparse.ArgumentParser(description="Closed-form cost/latency model.")
    parser.add_argument("--rho", type=float, default=RHO,
                        help="lambda*tau: fleet requests per cache lifetime (default 6).")
    parser.add_argument("--calls-per-batch", type=int, default=100,
                        help="classifications per batch for the cost table (default 100).")
    args = parser.parse_args()
    rho = args.rho
    calls = args.calls_per_batch

    # ---------- Section 1: latency ----------
    print("\n" + "=" * 70)
    print("1. LATENCY  -- V0 serial monolith vs V1 fork-join fleet (closed form)")
    print("=" * 70)
    print(f"l={L_LOAD} t_t={T_T} t_u={T_U} t_c={T_C} p={P} B={B_WARMUP}\n")
    hdr = f"{'K':>4} | {'L_serial (1)':>12} | {'L_fleet (2)':>12} | {'Speedup (3)':>11}"
    print(hdr); print("-" * len(hdr))
    lat_rows = []
    for k in BATCH_SIZES:
        ls, lf, s = l_serial(k), l_fleet(k), speedup(k)
        print(f"{k:>4} | {ls:>12.3f} | {lf:>12.3f} | {s:>10.2f}x")
        lat_rows.append({"K": k, "l_serial_s": round(ls, 4),
                         "l_fleet_s": round(lf, 4), "speedup": round(s, 4)})

    kc, kc_ceil = crossover_k()
    print(f"\n  Crossover (eq 4):    K* = {kc:.2f}  ->  first integer K = {kc_ceil}")
    print(f"  Asymptotic speedup (eq 5):  S_inf = {asymptotic_speedup():.2f}x "
          f"(> p={P}: serial also re-pays load + serial classify)")

    # ---------- Section 2: cache / cost ----------
    print("\n" + "=" * 70)
    print("2. CACHE EFFICIENCY -> COST  as classification fans out to C scopes")
    print("=" * 70)
    print(f"rho = lambda*tau = {rho}  |  c_f={C_FULL} c_h={C_CACHED} "
          f"l_f={L_FULL}ms l_h={L_CACHED}ms  |  {calls} calls/batch\n")
    hdr2 = (f"{'C':>3} | {'p_hit (6)':>9} | {'cost/call (7)':>13} | "
            f"{'Cost/batch (8)':>14} | {'lat/call (10)':>13}")
    print(hdr2); print("-" * len(hdr2))
    cost_rows = []
    base_cost = None
    for c in FLEET_SCOPES:
        ph = p_hit(c, rho)
        cc = cost_call(c, rho)
        batch = calls * cc
        lc = lat_call(c, rho)
        if base_cost is None:
            base_cost = batch
        print(f"{c:>3} | {ph:>9.3f} | {cc:>13.5f} | {batch:>14.4f} | {lc:>12.1f}")
        cost_rows.append({"scopes_C": c, "p_hit": round(ph, 4),
                          "cost_per_call": round(cc, 6),
                          "cost_per_batch": round(batch, 6),
                          "lat_per_call_ms": round(lc, 2)})

    worst = cost_rows[-1]
    print(f"\n  Fragmenting 1 -> {worst['scopes_C']} scopes (eq 9): "
          f"p_hit {cost_rows[0]['p_hit']:.3f} -> {worst['p_hit']:.3f}, "
          f"cost x{worst['cost_per_batch']/base_cost:.2f}, "
          f"call latency x{worst['lat_per_call_ms']/cost_rows[0]['lat_per_call_ms']:.2f}")
    print("  NOTE: K calls cost K calls regardless of execution parallelism p.")
    print("        Cost rises ONLY via cache fragmentation C. Keep C small (share")
    print("        one warm scope) while keeping p large -> latency win, no cost hit.")

    with open(CSV_LATENCY, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["K", "l_serial_s", "l_fleet_s", "speedup"])
        w.writeheader(); w.writerows(lat_rows)
    with open(CSV_COST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scopes_C", "p_hit", "cost_per_call",
                                           "cost_per_batch", "lat_per_call_ms"])
        w.writeheader(); w.writerows(cost_rows)
    print(f"\nWrote {CSV_LATENCY} and {CSV_COST}")


if __name__ == "__main__":
    main()
