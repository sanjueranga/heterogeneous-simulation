# Analytical Model — cost & latency of serial vs fork-join

Derivations behind `sims/model_analytical.py`. The scripts in `sims/` are
simulation checks of these equations; the agreement table is at the bottom.

## Symbols

| Symbol | Meaning |
|--------|---------|
| $K$ | clips in a batch |
| $\ell$ | model-load time (per clip in V0; amortized in V1) |
| $t_t, t_u, t_c$ | transcribe, cut, classify times |
| $p$ | heavy execution parallelism |
| $B$ | one-time broker + cold-start warmup |
| $C$ | number of independent **cache scopes** |
| $\lambda$ | classification request rate |
| $\tau$ | prompt-cache lifetime (TTL) |
| $\rho=\lambda\tau$ | fleet requests per cache lifetime |
| $c_f, c_h$ | full vs cached call cost ($c_h<c_f$) |
| $\ell_f, \ell_h$ | full vs cached call latency |

## 1. Latency

**V0 serial monolith** — one clip at a time, model reloaded each clip, no overlap:

$$L_{\text{serial}}(K) = K\,(\ell + t_t + t_u + t_c) \tag{1}$$

**V1 fork-join fleet** — $p$ heavy workers; classify is forked to the light pool so
it overlaps the cut, giving a per-clip critical path $t_t + \max(t_u,t_c)$; the
model load is paid once per worker child:

$$L_{\text{fleet}}(K) = B + \ell + \Big\lceil \tfrac{K}{p} \Big\rceil \big(t_t + \max(t_u,t_c)\big) \tag{2}$$

**Speedup:** $\;S(K) = L_{\text{serial}}(K)/L_{\text{fleet}}(K)$. (3)

**Crossover** (continuous root, dropping the ceiling): the smallest batch at which
the fleet wins,

$$K^\star = \frac{B + \ell}{(\ell + t_t + t_u + t_c) - \tfrac{1}{p}\big(t_t + \max(t_u,t_c)\big)} \tag{4}$$

**Asymptotic speedup** $K\to\infty$:

$$S_\infty = \frac{\ell + t_t + t_u + t_c}{\tfrac{1}{p}\big(t_t + \max(t_u,t_c)\big)} \tag{5}$$

$S_\infty$ can exceed $p$ because the serial path additionally re-pays $\ell$ every
clip and runs classify in series — both hidden by the fleet. With the repo-anchored
ratios ($\ell{=}.30,t_t{=}.10,t_u{=}.18,t_c{=}.12,p{=}4,B{=}.40$): $K^\star{=}2$,
$S_\infty{=}10\times$.

## 2. Cache efficiency → cost

**Clarifying the "$4\times$" intuition.** Running classification on $p=4$ workers
does *not* cost $4\times$ tokens: a batch of $K$ clips is still $K$ calls.
Parallelism raises cost only *indirectly*, by fragmenting the prompt cache.

**Derivation.** Classification calls share a long common prefix (system prompt +
taxonomy), so a call is cached iff a prior call hit the **same cache scope** within
the TTL $\tau$. The total request stream of rate $\lambda$ is split across $C$
independent scopes, each a Poisson stream of rate $\lambda/C$. For a Poisson
process, the probability that the preceding arrival on a scope lies within $\tau$ is
$1-e^{-(\lambda/C)\tau}$, hence

$$\boxed{\,p_{\text{hit}}(C) = 1 - e^{-\rho/C}\,}, \qquad \rho=\lambda\tau \tag{6}$$

- $C=1:\;p_{\text{hit}} = 1-e^{-\rho}$ (one warm scope; $\to 1$ when $\rho\gg1$).
- $C\gg\rho:\;p_{\text{hit}}\approx \rho/C \to 0$ (spread too thin, caches cold).

**Cost.** Per-call price mixes cached and full:

$$\text{cost}_{\text{call}}(C) = c_f - (c_f-c_h)\,p_{\text{hit}}(C) \tag{7}$$
$$\text{Cost}(K,C) = K\big[c_f - (c_f-c_h)\,p_{\text{hit}}(C)\big] \tag{8}$$

**Fragmentation penalty** (going from 1 to $C$ scopes):

$$\Delta\text{Cost}(C) = K(c_f-c_h)\big[e^{-\rho/C} - e^{-\rho}\big] \tag{9}$$

**Latency** behaves identically (cached calls have lower TTFT):

$$\ell_{\text{call}}(C) = \ell_f - (\ell_f-\ell_h)\,p_{\text{hit}}(C) \tag{10}$$

## 3. Design principle (the thesis, quantified)

Latency wants **high** $p$ (eq. 2, $L\sim 1/p$). Cost wants **low** $C$ (eq. 8).
These are *separate knobs*: $p$ is execution concurrency, $C$ is the count of cache
scopes. If the provider cache is content-keyed and **shared** across workers, then
$C=1$ for any $p$ — full latency win, zero cost penalty. The engineering lever is to
route classification so all workers share one warm cache scope (minimize $C$) while
maximizing execution parallelism $p$.

## 4. Model vs. simulation (validation)

`sims/model_analytical.py` (eq. 2) vs `measure_crossover.py` (Monte-Carlo), same constants:

| $K$ | $L_{\text{fleet}}$ analytical (2) | $L_{\text{fleet}}$ simulated | err |
|----:|----------------------------------:|-----------------------------:|----:|
| 1  | 0.980 | 0.998 | 1.8% |
| 2  | 0.980 | 1.001 | 2.1% |
| 5  | 1.260 | 1.290 | 2.3% |
| 10 | 1.540 | 1.573 | 2.1% |
| 20 | 2.100 | 2.150 | 2.3% |

Crossover $K^\star=2$ in both. The <3% gap is `time.sleep` scheduling overhead in
the simulator. Likewise `measure_cache_decay.py` is the Monte-Carlo form of eq. (6),
and `--calibrate` fits $\rho$ (equivalently $k$) to measured $(C, p_{\text{hit}})$.

> The stage times and cache constants in the sims are illustrative. The form of
> the model (eqs. 1-10) is what matters, plug in your own measurements.
