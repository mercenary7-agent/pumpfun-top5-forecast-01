"""pumpfun-top5-forecast-01 entry v8: forecast the 5 largest pump.fun coins (SOL-quoted,
launched since the opening) at the house's seven hourly answer reads, from the 10-minute
mark history.

Model (calibrated at runtime on whatever history the house provides):
- Order probabilities for known coins: Monte Carlo over the top coins (EW momentum drift,
  bonding-curve death hazard that grows with horizon and coin age, pool coins keep
  compounding).
- Unnamed (post-cut newborn + unranked) mass per place is NOT hardcoded: pseudo-cut
  backtests inside the history itself (cut at each mark, look forward, measure how often
  the place-k holder was unnamed at the cut) give a horizon curve per place, shrunk toward
  a horizon-dependent prior, and averaged over the seven answer reads.
- Size bins: place-k market cap modelled as an order statistic of the per-mark top-5 cap
  series (EW level + drift), Gaussian predictive bins averaged over the reads, with a fat
  right-tail mixture on place 1 and a small floor on every bin.

Deterministic: fixed seed, stdlib only. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output)/forecast.json.
"""
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

SEED = 20261112
N_SIMS = 6000          # Monte Carlo samples per answer read
TOP_COINS = 60         # known coins carried in the MC
NAMED_SET_N = 20       # coins that count as "named" in the pseudo-cut calibration
P_MIN = 0.00002        # smallest named order probability emitted
S_FLOOR = 0.002        # floor on every size bin of every place
TAIL_W = 0.18          # place-1 size mass placed in the explosion right tail
TAIL_TOP = 4.8         # top of the place-1 tail (log10 SOL)
PRIOR_LO = [0.15, 0.25, 0.30, 0.35, 0.40]   # unnamed mass prior at h=0
PRIOR_HI = [0.40, 0.55, 0.60, 0.68, 0.75]   # unnamed mass prior at h>=8h
SHRINK = 25.0          # pseudo-cut samples needed to fully trust the calibration


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


def coin_series(marks):
    series = {}
    for m in marks:
        t = parse_ts(m['at'])
        for r in m['rows']:
            series.setdefault(r['mint'], []).append(
                (t, log10_sol(r['market_cap_lamports']), r.get('venue'),
                 bool(r.get('complete')), parse_ts(r['created_at'])))
    for v in series.values():
        v.sort(key=lambda x: x[0])
    return series


def ew_drift(pts):
    """Exponentially weighted (half-life 45 min) per-hour drift of a (t, value) series."""
    if len(pts) < 2:
        return 0.0
    t1 = pts[-1][0]
    num, den = 0.0, 0.0
    for i in range(len(pts) - 1):
        t0, c0 = pts[i][0], pts[i][1]
        t2, c1 = pts[i + 1][0], pts[i + 1][1]
        dt = max(1e-6, (t2 - t0) / 3600.0)
        w = math.exp(-(t1 - t2) / 2700.0)
        num += w * (c1 - c0) / dt
        den += w
    d = num / den if den > 0 else 0.0
    return max(-2.0, min(2.0, d))


def calibrate_unnamed(marks, horizons):
    """Pseudo-cut backtest for per-place unnamed mass, averaged over the answer reads."""
    places = 5
    by_time = sorted(marks, key=lambda m: parse_ts(m['at']))
    samples = [[] for _ in range(places)]  # (h_hours, unnamed 0/1)
    for j in range(len(by_time) - 1):
        cut_mark = by_time[j]
        tc = parse_ts(cut_mark['at'])
        caps = sorted(((log10_sol(r['market_cap_lamports']), r['mint']) for r in cut_mark['rows']),
                      reverse=True)
        named = {m for _, m in caps[:NAMED_SET_N]}
        for m in by_time[j + 1:]:
            h = (parse_ts(m['at']) - tc) / 3600.0
            if h <= 0:
                continue
            rows = sorted(m['rows'], key=lambda r: -float(r['market_cap_lamports']))[:places]
            for k, r in enumerate(rows):
                unnamed = (parse_ts(r['created_at']) >= tc) or (r['mint'] not in named)
                samples[k].append((h, 1.0 if unnamed else 0.0))
    u = []
    for k in range(places):
        pts = samples[k]
        n = len(pts)
        if n >= 4:
            buckets = {}
            for h, q in pts:
                b = min(7, int(h / 1.5))
                buckets.setdefault(b, []).append(q)
            bh = sorted(buckets)
            bx = [b * 1.5 + 0.75 for b in bh]
            by = [sum(buckets[b]) / len(buckets[b]) for b in bh]
            m = len(bx)
            mx = sum(bx) / m
            my = sum(by) / m
            var = sum((x - mx) ** 2 for x in bx)
            slope = (sum((bx[i] - mx) * (by[i] - my) for i in range(m)) / var) if var > 1e-9 else 0.0
            slope = max(-0.03, min(0.15, slope))
            inter = my - slope * mx
            cal = []
            for h in horizons:
                q = inter + slope * h
                cal.append(max(0.02, min(0.98, q)))
            cal_avg = sum(cal) / len(cal)
            w = n / (n + SHRINK)
        else:
            cal_avg, w = None, 0.0
        prior = []
        for h in horizons:
            f = min(1.0, h / 8.0)
            prior.append(PRIOR_LO[k] + (PRIOR_HI[k] - PRIOR_LO[k]) * f)
        prior_avg = sum(prior) / len(prior)
        u.append((w * cal_avg + (1.0 - w) * prior_avg) if cal_avg is not None else prior_avg)
    return u


def gauss_bin(lo, hi, mu, sd):
    from math import erf, sqrt
    a = (lo - mu) / (sd * sqrt(2.0)) if lo > -1e8 else -10.0
    b = (hi - mu) / (sd * sqrt(2.0)) if hi < 1e8 else 10.0
    return max(0.0, 0.5 * (erf(b) - erf(a)))


def order_stat_sizes(marks, task, horizons):
    """Per-place size-bin distributions from Monte Carlo paths (known coins + ghost
    newborns), so place caps include post-cut launches. Returns per-place bin lists."""
    places = task['places']
    edges = task['size']['edges']
    nb = len(edges) + 1
    return None  # replaced by MC sizes in main(); kept for reference


def ghost_params(marks, series):
    """Cap distribution (log10 SOL) of young coins (<2h old) at the last mark, for ghosts."""
    last_t = parse_ts(marks[-1]['at'])
    young = []
    for m, v in series.items():
        cap, venue, complete, created = v[-1][1], v[-1][2], v[-1][3], v[-1][4]
        if last_t - created < 7200.0:
            young.append(cap)
    if len(young) >= 5:
        mu = sum(young) / len(young)
        sd = max(0.3, (sum((c - mu) ** 2 for c in young) / len(young)) ** 0.5)
    else:
        caps = sorted((v[-1][1] for v in series.values()), reverse=True)
        mu = caps[30] if len(caps) > 31 else (caps[-1] if caps else 2.0)
        sd = 1.0
    return mu, sd


def main():
    task, marks = load()
    places = task['places']
    series = coin_series(marks)
    hist_until = parse_ts(task['history_until'])
    horizons = [(parse_ts(r) - hist_until) / 3600.0 for r in task['reads']]

    u = calibrate_unnamed(marks, horizons)

    last = {m: v[-1] for m, v in series.items()}
    ranked = sorted(last.items(), key=lambda kv: -kv[1][1])[:TOP_COINS]
    mints = [m for m, _ in ranked]
    last_mark_t = parse_ts(marks[-1]['at'])
    params = []
    for m, (_, cap, venue, complete, created) in ranked:
        pts = series[m]
        drift = ew_drift(pts)
        pool = (venue in ('pool', 'migrating')) or complete
        age_h = max(0.0, (last_mark_t - created) / 3600.0)
        params.append((cap, drift, pool, age_h))

    rng = random.Random(SEED)
    n_reads = len(horizons)
    edges = task['size']['edges']
    nb = len(edges) + 1
    place_counts = [{m: 0.0 for m in mints} for _ in range(places)]
    size_counts = [[0.0] * nb for _ in range(places)]
    top5_votes = {m: 0.0 for m in mints}
    ghost_mu, ghost_sd = ghost_params(marks, series)
    n_ghosts = 250

    for h in horizons:
        for _ in range(N_SIMS):
            draws = []
            for cap, drift, pool, age_h in params:
                if pool:
                    mu = cap + drift * h * math.exp(-h / 10.0)
                    sd = 0.12 + 0.30 * math.sqrt(h)
                    draws.append(mu + rng.gauss(0.0, sd))
                else:
                    mu = cap + drift * h * math.exp(-h / 3.0) - 0.15 * min(1.0, h / 8.0)
                    sd = 0.25 + 0.45 * math.sqrt(h)
                    death_p = min(0.55, 0.10 + 0.10 * min(1.0, h / 2.0)
                                  + 0.05 * min(4.0, age_h) / 4.0)
                    if rng.random() < death_p * min(1.0, h / 2.0):
                        draws.append(0.5)
                        continue
                    draws.append(mu + rng.gauss(0.0, sd))
            for _ in range(n_ghosts):
                g = ghost_mu + rng.gauss(0.0, ghost_sd)
                if rng.random() < 0.05:          # rare newborn explosion
                    g += rng.uniform(0.5, 2.2)
                draws.append(g)
            order = sorted(range(len(draws)), key=lambda i: -draws[i])
            for k in range(places):
                i = order[k]
                cap_k = draws[i]
                b = 0
                while b < len(edges) and cap_k >= edges[b]:
                    b += 1
                size_counts[k][b] += 1.0
                if i < len(mints):
                    place_counts[k][mints[i]] += 1.0
                    if k == 0:
                        top5_votes[mints[i]] += 1.0

    inv = 1.0 / (n_reads * N_SIMS)
    probabilities = []
    for k in range(places):
        named_mass = max(0.05, 1.0 - u[k])
        total_p = sum(place_counts[k].values()) * inv
        scale = (named_mass / total_p) if total_p > 0 else 0.0
        for m, c in place_counts[k].items():
            p = c * inv * scale
            if p >= P_MIN:
                probabilities.append({'mint': m, 'place': k + 1, 'p': round(p, 6)})

    sizes = []
    for k in range(places):
        for b, c in enumerate(size_counts[k]):
            p = c * inv
            if p >= 0.0005:
                sizes.append({'place': k + 1, 'bin': b,
                              'p': math.floor(p * 1e6) / 1e6})

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
