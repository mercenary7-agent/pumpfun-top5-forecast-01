"""pumpfun-top5-forecast-01 entry: forecast the 5 largest pump.fun coins (SOL-quoted,
launched since the opening) at the house's seven hourly answer reads, from the 10-minute
mark history. Monte Carlo rank diffusion with per-coin momentum damped by horizon,
plus a flow of not-yet-born coins. Deterministic: fixed seed, stdlib only.

Reads $MAKEFUN_INPUT_DIR (else /input): task.json + history.ndjson.
Writes $MAKEFUN_OUTPUT_DIR (else /output)/forecast.json:
  {"probabilities": [{"mint","place","p"}...], "top5": [5 mints], "sizes": [{"place","bin","p"}...]}
"""
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

SEED = 20261112
N_SIMS = 8000          # Monte Carlo samples per answer read
GHOSTS = 200           # not-yet-born coins modelled at each answer read
TAU_CURVE = 3.0        # momentum decay (hours) for bonding-curve coins
TAU_POOL = 10.0        # for graduated (pool) coins
S0_CURVE, S1_CURVE = 0.25, 0.50   # sd = S0 + S1*sqrt(h)
S0_POOL, S1_POOL = 0.12, 0.32
DRIFT_CLIP = 2.0
NAMED_MASS = [0.65, 0.60, 0.60, 0.60, 0.60]  # named share per place; the rest is unnamed (newborns)
P_MIN = 0.00005        # smallest named order probability emitted
S_MIN = 0.000001       # smallest named size probability emitted


def parse_ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def log10_sol(lamports):
    return math.log10(float(lamports)) - 9.0


def load():
    with open(os.path.join(INPUT_DIR, 'task.json')) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, 'history.ndjson')) as f:
        for line in f:
            line = line.strip()
            if line:
                marks.append(json.loads(line))
    return task, [m for m in marks if m.get('ok') and isinstance(m.get('rows'), list)]


def coin_params(marks):
    """Per-mint series -> (last_logcap, drift per hour, is_pool)."""
    series = {}
    for m in marks:
        t = parse_ts(m['at'])
        for r in m['rows']:
            series.setdefault(r['mint'], []).append(
                (t, log10_sol(r['market_cap_lamports']), r.get('venue'), bool(r.get('complete'))))
    out = {}
    for mint, pts in series.items():
        pts.sort(key=lambda x: x[0])
        last_cap, venue, complete = pts[-1][1], pts[-1][2], pts[-1][3]
        # exponentially weighted drift over all sightings (half-life 45 min)
        drift = 0.0
        if len(pts) >= 2:
            t1 = pts[-1][0]
            num, den = 0.0, 0.0
            for i in range(len(pts) - 1):
                t0, c0 = pts[i][0], pts[i][1]
                t2, c1 = pts[i + 1][0], pts[i + 1][1]
                dt = max(1e-6, (t2 - t0) / 3600.0)
                w = math.exp(-(t1 - t2) / 2700.0)
                num += w * (c1 - c0) / dt
                den += w
            drift = num / den if den > 0 else 0.0
        drift = max(-DRIFT_CLIP, min(DRIFT_CLIP, drift))
        out[mint] = (last_cap, drift, (venue in ('pool', 'migrating')) or complete)
    return out


def main():
    task, marks = load()
    places = task['places']
    edges = task['size']['edges']
    coins = coin_params(marks)
    hist_until = parse_ts(task['history_until'])
    horizons = [(parse_ts(r) - hist_until) / 3600.0 for r in task['reads']]

    # keep the coins that can plausibly reach the top 5: top 80 by last cap
    ranked = sorted(coins.items(), key=lambda kv: -kv[1][0])[:80]
    mints = [m for m, _ in ranked]
    params = [v for _, v in ranked]

    # ghost (not-yet-born) coins: model the launch flow reaching the chart
    last_caps = sorted((v[0] for v in coins.values()), reverse=True)
    ghost_mu = last_caps[34] if len(last_caps) > 35 else (last_caps[-1] if last_caps else 2.0)
    ghost_sd = 1.0

    rng = random.Random(SEED)
    n_reads = len(horizons)
    place_counts = [{m: 0.0 for m in mints} for _ in range(places)]
    size_counts = [[0.0] * (len(edges) + 1) for _ in range(places)]
    top5_votes = {m: 0.0 for m in mints}

    for h in horizons:
        for _ in range(N_SIMS):
            draws = []
            kdraws = []
            for last_cap, drift, pool in params:
                tau = TAU_POOL if pool else TAU_CURVE
                s0, s1 = (S0_POOL, S1_POOL) if pool else (S0_CURVE, S1_CURVE)
                mu = last_cap + drift * h * math.exp(-h / tau)
                if not pool:
                    mu -= 0.15 * min(1.0, h / 8.0)  # bonding-curve coins fade on average
                sd = s0 + s1 * math.sqrt(h)
                d = mu + rng.gauss(0.0, sd)
                if not pool and rng.random() < 0.30 * min(1.0, h / 2.0):
                    d = 0.5  # bonding-curve coin dies: cap collapses
                draws.append(d)
                kdraws.append(last_cap + drift * h * 0.85 + rng.gauss(0.0, sd))  # size path: less damped
            for _ in range(GHOSTS):
                draws.append(ghost_mu + rng.gauss(0.0, ghost_sd))
            order = sorted(range(len(draws)), key=lambda i: -draws[i])
            korder = sorted(kdraws, reverse=True)
            for k in range(places):
                i = order[k]
                if k < len(korder):
                    size = korder[k]
                    b = 0
                    while b < len(edges) and size >= edges[b]:
                        b += 1
                    size_counts[k][b] += 1.0
                if i < len(mints):
                    place_counts[k][mints[i]] += 1.0
                    if k == 0:
                        top5_votes[mints[i]] += 1.0

    inv = 1.0 / (n_reads * N_SIMS)
    probabilities = []
    for k in range(places):
        mass = NAMED_MASS[k] if k < len(NAMED_MASS) else 0.45
        total_p = sum(place_counts[k].values()) * inv
        scale = (mass / total_p) if total_p > 0 else 0.0
        for m, c in place_counts[k].items():
            p = c * inv * scale
            if p >= P_MIN:
                probabilities.append({'mint': m, 'place': k + 1, 'p': round(p, 6)})
    sizes = []
    for k in range(places):
        for b, c in enumerate(size_counts[k]):
            p = c * inv
            if p >= S_MIN:
                sizes.append({'place': k + 1, 'bin': b, 'p': math.floor(p * 1e6) / 1e6})
    t5 = []
    seen = set()
    for m in sorted(mints, key=lambda m: -top5_votes[m]):
        if m not in seen:
            seen.add(m)
            t5.append(m)
        if len(t5) == places:
            break
    for m in mints:
        if len(t5) >= places:
            break
        if m not in seen:
            seen.add(m)
            t5.append(m)
    forecast = {'probabilities': probabilities, 'top5': t5, 'sizes': sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task['output']), 'w') as f:
        f.write(json.dumps(forecast) + '\n')


if __name__ == '__main__':
    main()
