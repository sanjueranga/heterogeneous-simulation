#!/usr/bin/env python3
"""
Sim for cost vs latency under queue-depth autoscaling.

A bursty batch of videos hits a pool of worker nodes (4 video slots per node).
Three policies are compared:

    static-1    always 1 node        (cheap, slow)
    static-max  always 10 nodes      (fast, expensive)
    autoscale   1..10 nodes, driven by queue depth, spot beyond the first
                on-demand node, with a cooldown between scaling actions

New nodes take a while to boot, which is why autoscaling lags a bit behind
static-max on latency and why one node is kept warm. Discrete-time, stdlib only,
prices are illustrative.

    python3 measure_autoscaling.py
    python3 measure_autoscaling.py --videos 480 --burst 600
"""

import argparse
import csv
import math
import os
import random

SLOTS_PER_NODE = 4
MIN_NODES = 1
MAX_NODES = 10
ON_DEMAND_BASE = 1        # first node is on-demand, the rest spot
COOLDOWN_S = 300
NODE_BOOT_S = 150         # launch -> serving

DEFAULT_VIDEOS = 240
DEFAULT_BURST_S = 300     # arrival window, smaller = burstier
SERVICE_MEAN_S = 90.0     # per-video heavy task time
SERVICE_JITTER_S = 20.0
# long enough for even static-1 to drain (240 x 90s / 4 slots ~ 90 min), so p95
# is real for every policy and all fleets are billed over the same window
SIM_DURATION_S = 9000
TICK_S = 5

# $ per node-hour. On-demand is roughly a 4 vCPU / 8 GiB compute instance list
# price, spot is a typical discount. Swap in your own numbers.
PRICE_ON_DEMAND = 0.153
PRICE_SPOT = 0.060

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim", "autoscaling.csv")


def make_workload(n_videos, burst_s, seed=42):
    rng = random.Random(seed)
    arrivals = sorted(rng.uniform(0, burst_s) for _ in range(n_videos))
    services = [max(5.0, rng.gauss(SERVICE_MEAN_S, SERVICE_JITTER_S)) for _ in range(n_videos)]
    return arrivals, services


def _percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def simulate(policy, arrivals, services, n_videos):
    """
    Tick-based simulation. Returns metrics dict.
    A 'node' dict: {ready_at, kind}. It is serving once t >= ready_at.
    """
    # Seed initial nodes (ready at t=0).
    def new_node(kind, ready_at):
        return {"ready_at": ready_at, "kind": kind}

    if policy == "static-1":
        nodes = [new_node("on_demand", 0)]
        fixed = True
    elif policy == "static-max":
        nodes = [new_node("on_demand" if i < ON_DEMAND_BASE else "spot", 0)
                 for i in range(MAX_NODES)]
        fixed = True
    else:  # autoscale
        nodes = [new_node("on_demand", 0)]
        fixed = False

    arr_idx = 0
    queue = []                      # list of (video_index, arrival_time)
    running = []                    # list of finish_times
    completions = {}                # video_index -> completion_time
    node_seconds = {"on_demand": 0.0, "spot": 0.0}
    last_scale_t = -COOLDOWN_S
    max_nodes_used = len(nodes)

    arrivals_idx = sorted(range(n_videos), key=lambda i: arrivals[i])

    t = 0
    while t <= SIM_DURATION_S:
        # 1. Arrivals up to t.
        while arr_idx < n_videos and arrivals[arrivals_idx[arr_idx]] <= t:
            vi = arrivals_idx[arr_idx]
            queue.append((vi, arrivals[vi]))
            arr_idx += 1

        # 2. Completions.
        still = []
        for vi, fin in running:
            if fin <= t:
                completions[vi] = fin
            else:
                still.append((vi, fin))
        running = still

        # 3. Accrue cost for every node that exists this tick (booting nodes
        #    cost money too).
        for nd in nodes:
            node_seconds[nd["kind"]] += TICK_S

        ready_nodes = sum(1 for nd in nodes if t >= nd["ready_at"])
        free_slots = ready_nodes * SLOTS_PER_NODE - len(running)

        # 4. Scaling (autoscale only), gated by cooldown.
        if not fixed and (t - last_scale_t) >= COOLDOWN_S:
            backlog = len(queue) + len(running)
            target = max(MIN_NODES, min(MAX_NODES, math.ceil(backlog / SLOTS_PER_NODE)))
            cur = len(nodes)
            if target > cur:
                for i in range(cur, target):
                    kind = "on_demand" if i < ON_DEMAND_BASE else "spot"
                    nodes.append(new_node(kind, t + NODE_BOOT_S))
                last_scale_t = t
            elif target < cur and not queue:
                # Scale in: drop the newest (spot) nodes first, keep the base.
                remove = cur - max(target, MIN_NODES)
                if remove > 0:
                    nodes = nodes[:cur - remove]
                    last_scale_t = t
            max_nodes_used = max(max_nodes_used, len(nodes))

        # 5. Dispatch queued videos to free slots.
        ready_nodes = sum(1 for nd in nodes if t >= nd["ready_at"])
        free_slots = ready_nodes * SLOTS_PER_NODE - len(running)
        while queue and free_slots > 0:
            vi, _arr = queue.pop(0)
            running.append((vi, t + services[vi]))
            free_slots -= 1

        # Run the full horizon: static fleets keep paying while idle; autoscale
        # scales back to the base node (scale-to-near-zero) on the idle tail.
        t += TICK_S

    # Flush late completions.
    for vi, fin in running:
        completions[vi] = fin

    latencies = [completions[i] - arrivals[i] for i in range(n_videos) if i in completions]
    on_demand_hours = node_seconds["on_demand"] / 3600.0
    spot_hours = node_seconds["spot"] / 3600.0
    cost = on_demand_hours * PRICE_ON_DEMAND + spot_hours * PRICE_SPOT
    total_hours = on_demand_hours + spot_hours

    return {
        "policy": policy,
        "p95_latency_min": _percentile(latencies, 95) / 60.0,
        "mean_latency_min": (sum(latencies) / len(latencies) / 60.0) if latencies else 0.0,
        "completed": len(latencies),
        "node_hours": total_hours,
        "spot_frac": (spot_hours / total_hours) if total_hours else 0.0,
        "max_nodes": max_nodes_used,
        "cost": cost,
    }


def main():
    parser = argparse.ArgumentParser(description="Cost vs latency under autoscaling.")
    parser.add_argument("--videos", type=int, default=DEFAULT_VIDEOS)
    parser.add_argument("--burst", type=int, default=DEFAULT_BURST_S,
                        help="Arrival window in seconds (smaller = burstier).")
    args = parser.parse_args()

    arrivals, services = make_workload(args.videos, args.burst)

    print("\nAutoscaling cost vs latency (simulated)")
    print(f"Workload: {args.videos} videos arriving over {args.burst}s burst | "
          f"{SLOTS_PER_NODE} slots/node | boot {NODE_BOOT_S}s | cooldown {COOLDOWN_S}s")
    print(f"Prices ($/node-hr): on-demand {PRICE_ON_DEMAND}, spot {PRICE_SPOT}\n")

    header = (f"{'Policy':<11} | {'P95 lat (min)':>13} | {'Mean (min)':>10} | "
              f"{'Node-hrs':>8} | {'Spot %':>6} | {'Max N':>5} | {'Cost ($)':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for policy in ("static-1", "static-max", "autoscale"):
        m = simulate(policy, arrivals, services, args.videos)
        print(f"{m['policy']:<11} | {m['p95_latency_min']:>13.1f} | "
              f"{m['mean_latency_min']:>10.1f} | {m['node_hours']:>8.2f} | "
              f"{m['spot_frac']*100:>5.0f}% | {m['max_nodes']:>5} | {m['cost']:>9.3f}")
        rows.append({
            "policy": m["policy"],
            "p95_latency_min": round(m["p95_latency_min"], 3),
            "mean_latency_min": round(m["mean_latency_min"], 3),
            "node_hours": round(m["node_hours"], 4),
            "spot_frac": round(m["spot_frac"], 4),
            "max_nodes": m["max_nodes"],
            "cost": round(m["cost"], 4),
        })

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "policy", "p95_latency_min", "mean_latency_min",
            "node_hours", "spot_frac", "max_nodes", "cost"])
        writer.writeheader()
        writer.writerows(rows)

    sm = next(r for r in rows if r["policy"] == "static-max")
    au = next(r for r in rows if r["policy"] == "autoscale")
    if sm["cost"] > 0:
        print(f"\nAutoscale vs static-max: P95 latency "
              f"{au['p95_latency_min']:.1f} vs {sm['p95_latency_min']:.1f} min, "
              f"cost ${au['cost']:.3f} vs ${sm['cost']:.3f} "
              f"({(1 - au['cost']/sm['cost'])*100:.0f}% cheaper).")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
