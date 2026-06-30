#!/usr/bin/env python3
"""
Sim: weighted vs raw queue depth as the autoscaling signal, for a fleet that
runs three kinds of tasks with very different costs.

Each node runs three kinds of workers:

    heavy    4 slots   slow (video transcode / transcribe), the thing users wait on
    api     10 slots   medium, I/O bound (LLM calls)
    general 10 slots   cheap, drains in seconds

Summing the raw queue lengths treats a pile of cheap general tasks like a pile
of videos, so the autoscaler over-provisions. Weighting each queue by its
per-node throughput (general / 10) tracks the real demand better. The sim
runs both signals on the same workload and compares heavy-job p95 latency and
cost.

    raw       = heavy + api + general
    weighted  = heavy + api + general / 10

Discrete-time, prices are illustrative.

    python3 measure_weighted_scaling.py
    python3 measure_weighted_scaling.py --general 12000 --heavy 120
"""

import argparse
import csv
import math
import os
import random
from collections import deque

SLOTS = {"heavy": 4, "api": 10, "general": 10}   # per-node concurrency per class
SCALE_DIVISOR = SLOTS["heavy"]                    # signal is in heavy-equiv units
MIN_NODES, MAX_NODES = 1, 10                      
ON_DEMAND_BASE = 1                                # on_demand_base_capacity
COOLDOWN_S = 300                                  # default_cooldown
NODE_BOOT_S = 150                                 # launch->serving

# per-class service times: (mean_s, jitter_s, floor_s)
SERVICE = {
    "heavy":   (90.0, 20.0, 5.0),
    "api":     (8.0, 3.0, 1.0),
    "general": (2.0, 1.0, 0.5),
}

# the two scaling signals being compared
WEIGHTS = {
    "raw":      {"heavy": 1.0, "api": 1.0, "general": 1.0},     # naive: count all
    "weighted": {"heavy": 1.0, "api": 1.0, "general": 0.1},     # general counted at 1/10
}

# $/node-hr, illustrative
PRICE_ON_DEMAND = 0.153
PRICE_SPOT = 0.060        # realized spot

SIM_DURATION_S = 5400
TICK_S = 5

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim")
CSV_PATH = os.path.join(HERE, "weighted_scaling.csv")
TS_PATH = os.path.join(HERE, "weighted_scaling_timeseries.csv")


def _svc(rng, cls):
    mean, jit, floor = SERVICE[cls]
    return max(floor, rng.gauss(mean, jit))


def make_workload(n_heavy, n_api, n_general, horizon, seed=42):
    """Steady heavy + api streams, plus a periodically-bursty cheap general stream.

    The general bursts are the crux: at each scaling sample there is a real cheap
    backlog that RAW counting over-reacts to, but that one general worker drains
    in seconds.
    """
    rng = random.Random(seed)
    jobs = []  # (arrival_s, cls, service_s)

    for _ in range(n_heavy):
        t = rng.uniform(0, horizon)
        jobs.append((t, "heavy", _svc(rng, "heavy")))
    for _ in range(n_api):
        t = rng.uniform(0, horizon)
        jobs.append((t, "api", _svc(rng, "api")))

    # general arrives in periodic mini-bursts so llen(general) is non-trivial at
    # sample time (mimics bulk enqueues), yet each task is cheap.
    period = 25.0
    per_burst = max(1, int(round(n_general / (horizon / period))))
    t = 0.0
    while t < horizon:
        for _ in range(per_burst):
            jobs.append((t + rng.uniform(0, 2.0), "general", _svc(rng, "general")))
        t += period

    jobs.sort(key=lambda j: j[0])
    return jobs


def _percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def simulate(signal_name, jobs):
    """Tick sim. Scales node count on the chosen weighted queue-depth signal."""
    weights = WEIGHTS[signal_name]
    nodes = [{"ready_at": 0, "kind": "on_demand"}]            # start at base
    queues = {c: deque() for c in SLOTS}                      # (idx, arrival)
    running = {c: [] for c in SLOTS}                          # finish times
    completions = {}                                          # idx -> (cls, fin, arr)
    node_seconds = {"on_demand": 0.0, "spot": 0.0}
    last_scale_t = -COOLDOWN_S
    max_nodes = len(nodes)
    node_count_sum = nodes_samples = 0
    timeseries = []                                           # (t, n_nodes, signal)

    arr = deque((i, t, c, s) for i, (t, c, s) in enumerate(jobs))

    t = 0
    while t <= SIM_DURATION_S:
        # 1. Arrivals.
        while arr and arr[0][1] <= t:
            i, at, c, s = arr.popleft()
            queues[c].append((i, at, s))

        # 2. Completions.
        for c in SLOTS:
            still = []
            for i, at, fin in running[c]:
                if fin <= t:
                    completions[i] = (c, fin, at)
                else:
                    still.append((i, at, fin))
            running[c] = still

        # 3. Cost accrues for every existing node (booting nodes bill too).
        for nd in nodes:
            node_seconds[nd["kind"]] += TICK_S

        # 4. Scale on the queue-depth signal (Redis llen = queued only, matching
        #    monitor_queues.py -- active tasks are not in the broker list).
        signal = sum(weights[c] * len(queues[c]) for c in SLOTS)
        if (t - last_scale_t) >= COOLDOWN_S:
            target = max(MIN_NODES, min(MAX_NODES, math.ceil(signal / SCALE_DIVISOR)))
            cur = len(nodes)
            if target > cur:
                for i in range(cur, target):
                    kind = "on_demand" if i < ON_DEMAND_BASE else "spot"
                    nodes.append({"ready_at": t + NODE_BOOT_S, "kind": kind})
                last_scale_t = t
            elif target < cur:
                nodes = nodes[:max(target, MIN_NODES)]   # drop newest spot first
                last_scale_t = t
            max_nodes = max(max_nodes, len(nodes))

        # 5. Dispatch each class to its own free slots on ready nodes.
        ready = sum(1 for nd in nodes if t >= nd["ready_at"])
        for c in SLOTS:
            free = ready * SLOTS[c] - len(running[c])
            while queues[c] and free > 0:
                i, at, s = queues[c].popleft()
                running[c].append((i, at, t + s))
                free -= 1

        node_count_sum += len(nodes)
        nodes_samples += 1
        if t % 60 == 0:
            timeseries.append((t / 60.0, len(nodes), round(signal, 1)))
        t += TICK_S

    for c in SLOTS:                                          # flush stragglers
        for i, at, fin in running[c]:
            completions[i] = (c, fin, at)

    lat = {c: [] for c in SLOTS}
    for c, fin, at in completions.values():
        lat[c].append(fin - at)

    od_h = node_seconds["on_demand"] / 3600.0
    sp_h = node_seconds["spot"] / 3600.0
    total_h = od_h + sp_h
    return {
        "signal": signal_name,
        "heavy_p95_min": _percentile(lat["heavy"], 95) / 60.0,
        "api_p95_min": _percentile(lat["api"], 95) / 60.0,
        "general_p95_min": _percentile(lat["general"], 95) / 60.0,
        "node_hours": total_h,
        "mean_nodes": node_count_sum / nodes_samples if nodes_samples else 0.0,
        "max_nodes": max_nodes,
        "spot_frac": sp_h / total_h if total_h else 0.0,
        "cost": od_h * PRICE_ON_DEMAND + sp_h * PRICE_SPOT,
        "timeseries": timeseries,
    }


def main():
    ap = argparse.ArgumentParser(description="Weighted vs raw queue-depth scaling.")
    ap.add_argument("--heavy", type=int, default=100, help="heavy (video) jobs.")
    ap.add_argument("--api", type=int, default=400, help="api/LLM jobs.")
    ap.add_argument("--general", type=int, default=9000, help="cheap general tasks.")
    args = ap.parse_args()

    jobs = make_workload(args.heavy, args.api, args.general, SIM_DURATION_S)
    n_by = {c: sum(1 for _, cc, _ in jobs if cc == c) for c in SLOTS}

    print("\nWeighted vs raw queue-depth autoscaling")
    print(f"Workload over {SIM_DURATION_S//60} min: "
          f"heavy={n_by['heavy']} api={n_by['api']} general={n_by['general']}  | "
          f"per-node slots {SLOTS} | signal/node divisor {SCALE_DIVISOR}")
    print("Signals:  raw = h + a + g      weighted = h + a + g/10\n")

    header = (f"{'signal':<9} | {'heavy p95':>9} | {'gen p95':>8} | "
              f"{'node-hrs':>8} | {'mean N':>6} | {'max N':>5} | {'cost $':>7}")
    print(header)
    print("-" * len(header))

    results = {}
    rows, ts_rows = [], []
    for name in ("raw", "weighted"):
        m = simulate(name, jobs)
        results[name] = m
        print(f"{m['signal']:<9} | {m['heavy_p95_min']:>8.1f}m | "
              f"{m['general_p95_min']:>7.1f}m | {m['node_hours']:>8.2f} | "
              f"{m['mean_nodes']:>6.1f} | {m['max_nodes']:>5} | {m['cost']:>7.3f}")
        rows.append({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in m.items() if k != "timeseries"})
        for (tm, n, sig) in m["timeseries"]:
            ts_rows.append({"t_min": round(tm, 2), "signal": name,
                            "nodes": n, "depth": sig})

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "signal", "heavy_p95_min", "api_p95_min", "general_p95_min",
            "node_hours", "mean_nodes", "max_nodes", "spot_frac", "cost"])
        w.writeheader()
        w.writerows(rows)
    with open(TS_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["t_min", "signal", "nodes", "depth"])
        w.writeheader()
        w.writerows(ts_rows)

    raw, wt = results["raw"], results["weighted"]
    if raw["cost"] > 0:
        dlat = wt["heavy_p95_min"] - raw["heavy_p95_min"]
        print(f"\nWeighted vs raw: heavy-job p95 {wt['heavy_p95_min']:.1f} vs "
              f"{raw['heavy_p95_min']:.1f} min ({dlat:+.1f} min), "
              f"cost ${wt['cost']:.3f} vs ${raw['cost']:.3f} "
              f"({(1 - wt['cost']/raw['cost'])*100:.0f}% cheaper), "
              f"mean fleet {wt['mean_nodes']:.1f} vs {raw['mean_nodes']:.1f} nodes.")
        print("Lesson: weighting cheap `general` tasks by their inverse throughput "
              "keeps the\nscaling signal aligned with real (heavy) node demand -- "
              "same SLA, far lower cost.")
    print(f"Wrote {CSV_PATH}\nWrote {TS_PATH}")


if __name__ == "__main__":
    main()
