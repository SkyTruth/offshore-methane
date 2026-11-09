"""Dark-run (Bernoulli CUSUM with zero off-state detection rate) scan.

For each site, passes in time order; a dark run is a maximal sequence with no
clear detection (D) and no cloudy detection (C). Its score is
S = sum over clear non-detections (Z) of -log(1 - p_j), with p_j the site's
baseline per-pass detection probability from clear passes before the run,
shifted on the logit scale by view zone and satellite (s01 offsets).
Under H0 (still lit, independent passes) P(run) = exp(-S).
Null calibration: shuffle D/Z labels within site x zone (C and U fixed).
"""
# %%
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import CACHE, passes, sites  # noqa: E402

ZOFF = {0: 0.0, 1: -0.474, 2: -1.031}
SOFF = {"NOAA-20": 0.0, "NOAA-21": -0.111, "SNPP": -0.425}
MIN_BASE_CLEAR, MIN_BASE_DET, MIN_BASE_F = 20, 6, 0.3
H = [np.log(1e3), np.log(1e4), np.log(1e5)]


def logit(x):
    return np.log(x / (1 - x))


def expit(x):
    return 1 / (1 + np.exp(-x))


def scan_site(out, off, night):
    """Return list of runs: (start_idx, end_idx, S, n_clear, base_f, n_base_clear)."""
    isD = out == "D"
    isZ = out == "Z"
    lit = isD | (out == "C")
    cumD = np.cumsum(isD)
    cumC = np.cumsum(isD | isZ)
    runs = []
    n = len(out)
    i = 0
    while i < n:
        if lit[i]:
            i += 1
            continue
        j = i
        while j < n and not lit[j]:
            j += 1
        # run i..j-1 ; baseline = passes before i
        nb = cumC[i - 1] if i > 0 else 0
        db = cumD[i - 1] if i > 0 else 0
        if nb >= MIN_BASE_CLEAR and db >= MIN_BASE_DET and db / nb >= MIN_BASE_F:
            f = min(db / nb, 0.98)
            z = isZ[i:j]
            if z.any():
                # centre offsets on baseline mean geometry
                base_off = off[:i][(isD | isZ)[:i]].mean()
                pj = expit(logit(f) + off[i:j][z] - base_off)
                S = -np.log1p(-pj).sum()
                runs.append((i, j - 1, S, int(z.sum()), f, int(nb)))
        i = j
    return runs


def scan(p, rng=None):
    rows = []
    for sid, g in p.groupby("site_id", sort=False):
        out = g.out.to_numpy().copy()
        if (out == "D").sum() < MIN_BASE_DET:
            continue
        if rng is not None:
            for zv in (0, 1, 2):
                m = np.isin(out, ["D", "Z"]) & (g.zone.to_numpy() == zv)
                out[m] = rng.permutation(out[m])
        off = g.off.to_numpy()
        for i, j, S, nz, f, nb in scan_site(out, off, g.night.to_numpy()):
            rows.append((sid, g.t.iloc[i], g.t.iloc[j], g.night.iloc[i],
                         g.night.iloc[j], S, nz, f, nb, j == len(out) - 1))
    return pd.DataFrame(rows, columns=["site_id", "t0", "t1", "night0", "night1", "S",
                                       "n_clear", "base_f", "n_base", "ongoing"])


def main(n_perm=20):
    p = passes()
    p["off"] = p.zone.map(ZOFF) + p.satellite.map(SOFF)
    runs = scan(p)
    runs.to_pickle(CACHE / "runs.pkl")
    print("runs scored", len(runs), "sites", runs.site_id.nunique())
    rng = np.random.default_rng(0)
    null = []
    for k in range(n_perm):
        r = scan(p, rng)
        null.append([(r.S >= h).sum() for h in H] + [r[r.S >= h].site_id.nunique() for h in H])
    null = np.array(null)
    for k, h in enumerate(H):
        obs = runs[runs.S >= h]
        print(f"h=ln(1/{np.exp(h):.0e}): observed runs {len(obs)} at {obs.site_id.nunique()} "
              f"sites; permutation null runs mean {null[:, k].mean():.1f} "
              f"(max {null[:, k].max()}), sites mean {null[:, 3 + k].mean():.1f}")
    # max clear-run length distribution vs null for steady sites
    return runs


if __name__ == "__main__":
    main()
