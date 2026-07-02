"""Experiment C: prompt-cache hit rate and cost vs number of cache scopes.

For each C in --scopes, N scene classifications are spread round-robin over C
scopes (a different marker in the cached prefix each), at --rate calls/sec. A call
is a hit when cache_read_input_tokens > 0. Writes p_hit(C), avg latency and cost
per batch to cache_decay.csv, and with --headline also a caching off vs on
comparison at C=1 to cache_headline.csv.

Scene text comes from the DB (run a v1 batch first), with a small built-in
sample as fallback. N is capped so a typo can't burn money.

  python manage.py cost_sweep --scopes 1 2 4 8 16 --n 200 --rate 3
"""
import math
import os
import time
import itertools
from django.core.management.base import BaseCommand

from pipeline import classify, instrument
from pipeline.models import Scene

_FALLBACK = [
    "Ever wonder why the simplest habit is the one nobody keeps?",
    "It started as a weekend experiment and turned into everything.",
    "Here's the part most people get wrong about staying consistent.",
    "Watch what happens when you change just one variable.",
    "Nine out of ten people told us the same surprising thing.",
    "Sure, it takes longer up front — but look at the payoff.",
    "The secret wasn't working harder, it was working slower.",
    "So now the whole process runs while you sleep.",
    "If you take one thing away, let it be this.",
    "Give it a try and see for yourself.",
]

_DECAY_HEADER = ["instances", "p_hit", "p_hit_law", "avg_latency_ms", "avg_cost_per_batch"]
_HEAD_HEADER = ["config", "p_hit", "cost", "pct_saving"]
_CONC_HEADER = ["threads", "instances", "p_hit", "avg_latency_ms", "avg_cost_per_batch"]


def _corpus(n):
    texts = list(Scene.objects.exclude(text="").values_list("text", flat=True))
    if not texts:
        texts = _FALLBACK
    return list(itertools.islice(itertools.cycle(texts), n))


class Command(BaseCommand):
    help = "Experiment C: measured p_hit(C) and cost, plus off-vs-on headline."

    def add_arguments(self, p):
        p.add_argument("--scopes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
        p.add_argument("--n", type=int, default=200)
        p.add_argument("--n-per-scope", type=int, default=0,
                       help="if set, N for each C = n_per_scope * C so every scope "
                            "gets about that many calls (the first call per scope is "
                            "always a cold write)")
        p.add_argument("--rate", type=float, default=3.0, help="calls/sec overall")
        p.add_argument("--threads", type=int, nargs="+", default=[1],
                       help="concurrent classify calls per batch, can take several "
                            "values (--threads 1 2 4 8). With >1 thread the first wave "
                            "races the cache write and misses. Anything other than a "
                            "plain --threads 1 goes to cache_concurrency.csv")
        p.add_argument("--model", default=os.environ.get("CLASSIFY_MODEL", "claude-haiku-4-5"))
        p.add_argument("--headline", action="store_true",
                       help="also run caching off-vs-on at C=1")

    def handle(self, *a, **o):
        os.environ["CLASSIFY_MODE"] = "injection"
        # force the real backend, a leftover CLASSIFY_BACKEND=mock from the
        # memory runs would turn every call into a free no-op
        os.environ["CLASSIFY_BACKEND"] = "real"
        os.environ["CLASSIFY_THREADS"] = str(o["threads"])
        os.environ["COST_RUN_SALT"] = str(int(time.time()))  # fresh caches this run
        sleep = 1.0 / o["rate"] if o["rate"] > 0 else 0.0

        conc_mode = len(o["threads"]) > 1 or max(o["threads"]) > 1

        for C in o["scopes"]:
            os.environ["CACHE_SCOPES"] = str(C)
            n_c = o["n_per_scope"] * C if o["n_per_scope"] else o["n"]
            corpus = _corpus(n_c)

            for T in o["threads"]:
                os.environ["CLASSIFY_THREADS"] = str(T)
                self.stdout.write(f"[C={C} threads={T}] {n_c} calls "
                                  f"(~{n_c // C}/scope) @ {o['rate']}/s")

                def _fire(i, T=T):
                    # scope id depends on C and T, so every sweep point starts
                    # with its own cold cache
                    sid = (T - 1) * 100000 + C * 1000 + (i % C)
                    _, u = classify.classify_scene(corpus[i], scope_id=sid,
                                                   caching=True, model=o["model"])
                    instrument.log_classify(u, scene_id="")
                    return u

                usages = []
                if T <= 1:
                    for i in range(len(corpus)):
                        usages.append(_fire(i))
                        if sleep:
                            time.sleep(sleep)
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=T) as ex:
                        futs = [ex.submit(_fire, i) for i in range(len(corpus))]
                        for f in as_completed(futs):
                            usages.append(f.result())

                # skip each scope's first call, it's always a cold write
                seen = set()
                measured = []
                for u in sorted(usages, key=lambda u: u["arrival_ts"]):
                    if u["scope_id"] in seen:
                        measured.append(u)
                    seen.add(u["scope_id"])
                hits = sum(1 for u in measured if u["cache_hit"])
                p_hit = hits / len(measured) if measured else 0.0
                lat = sum(u["latency_ms"] for u in usages)
                cost = sum(u["cost_usd"] for u in usages)  # cost counts every call

                if conc_mode:
                    instrument._append("cache_concurrency.csv", _CONC_HEADER, {
                        "threads": T, "instances": C, "p_hit": round(p_hit, 4),
                        "avg_latency_ms": round(lat / n_c, 2),
                        "avg_cost_per_batch": round(cost, 6),
                    })
                else:
                    instrument._append("cache_decay.csv", _DECAY_HEADER, {
                        "instances": C, "p_hit": round(p_hit, 4),
                        "p_hit_law": "",  # filled in later by probes/fit_cache.py
                        "avg_latency_ms": round(lat / n_c, 2),
                        "avg_cost_per_batch": round(cost, 6),
                    })
                self.stdout.write(f"[C={C} threads={T}] p_hit={p_hit:.3f} "
                                  f"cost=${cost:.4f}")

        if o["headline"]:
            self._headline(corpus[: min(50, len(corpus))], o["model"])

    def _headline(self, corpus, model):
        os.environ["CACHE_SCOPES"] = "1"
        rows = []
        for cfg, caching in (("caching_off", False), ("caching_on_warm", True)):
            cost = 0.0
            hits = 0
            for i, text in enumerate(corpus):
                _, u = classify.classify_scene(text, scope_id=0, caching=caching, model=model)
                instrument.log_classify(u, scene_id="")
                cost += u["cost_usd"]
                hits += 1 if u["cache_hit"] else 0
            rows.append((cfg, hits / len(corpus), cost))
        off = next(c for n, _, c in rows if n == "caching_off")
        for name, ph, cost in rows:
            saving = 0.0 if name == "caching_off" else round(100 * (off - cost) / off, 2)
            instrument._append("cache_headline.csv", _HEAD_HEADER, {
                "config": name, "p_hit": round(ph, 4),
                "cost": round(cost, 6), "pct_saving": saving,
            })
        self.stdout.write(f"[headline] off=${off:.4f} -> on saving printed to cache_headline.csv")
