"""Round 2 run-score audit; reads Opus caches without changing them.

Run from nightly-flaring/ with PYTHONDONTWRITEBYTECODE=1 .venv/bin/python.
Block shuffles are dependence sensitivity checks, not labelled benign nulls.
"""

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from s02_runs import H, SOFF, ZOFF, scan_site

OUT = Path("outputs/flareout_research/scripts_sol")
CACHE = Path("outputs/flareout_research/scripts_opus/cache")


def count(groups, rng=None, block=None):
    counts = np.zeros(3, dtype=int)
    sites = np.zeros(3, dtype=int)
    for sid, out, off, zone, night in groups:
        if rng is not None:
            if block is None:
                out = out.copy()
                for z in range(3):
                    m = np.isin(out, ["D", "Z"]) & (zone == z)
                    out[m] = rng.permutation(out[m])
            else:
                # Reorder contiguous calendar-night bundles; preserve within-night
                # outcome/geometry pairs and all cloudy detections and unknowns.
                day = (pd.to_datetime(night) - pd.Timestamp("2026-08-09")).days
                label = np.asarray(day) // block
                units = [np.flatnonzero(label == b) for b in np.unique(label)]
                order = rng.permutation(len(units))
                idx = np.concatenate([units[b] for b in order])
                out, off = out[idx], off[idx]
        scores = np.array([r[2] for r in scan_site(out, off, night)])
        c = np.array([(scores >= h).sum() for h in H])
        counts += c
        sites += c > 0
    return counts, sites


def main():
    start = time.time()
    p = pd.read_pickle(CACHE / "passes.pkl")
    p["off"] = p.zone.map(ZOFF) + p.satellite.map(SOFF)
    nd = p[p.out == "D"].groupby("site_id").size()
    p = p[p.site_id.isin(nd[nd >= 6].index)].copy()
    groups = [(sid, g.out.to_numpy(), g.off.to_numpy(), g.zone.to_numpy(),
               g.night.to_numpy()) for sid, g in p.groupby("site_id", sort=False)]
    observed, nsites = count(groups)
    result = {"observed_runs": observed.tolist(), "observed_sites": nsites.tolist(),
              "eligible_ever6_sites": len(groups), "passes": len(p), "nulls": {}}
    print("Observed", result, flush=True)
    for block, reps in [(None, 20), (1, 100), (3, 100), (7, 100)]:
        rng = np.random.default_rng(0 if block is None else 62300 + block)
        totals = np.array([count(groups, rng, block)[0] for _ in range(reps)])
        name = "within_zone_pass_shuffle" if block is None else f"{block}_calendar_night_blocks"
        result["nulls"][name] = {"replicates": reps, "mean": totals.mean(axis=0).tolist(),
                                 "q95": np.quantile(totals, .95, axis=0).tolist(),
                                 "min": totals.min(axis=0).tolist(),
                                 "max": totals.max(axis=0).tolist()}
        print(name, result["nulls"][name], "elapsed", time.time()-start, flush=True)
    # Same fixed offsets, changing only outcome-dependent cloud eligibility.
    p["out_raw"] = np.select([p.detected & p.cloud_mask.isin([0, 1]),
                              ~p.detected & p.cloud_mask.isin([0, 1]),
                              p.detected], ["D", "Z", "C"], "U")
    raw_groups = [(sid, g.out_raw.to_numpy(), g.off.to_numpy(), g.zone.to_numpy(),
                   g.night.to_numpy()) for sid, g in p.groupby("site_id", sort=False)]
    result["raw_cloud01_fixed_offsets"] = {"runs": count(raw_groups)[0].tolist(),
                                          "sites": count(raw_groups)[1].tolist()}
    result["strict_override_detections"] = int(((p.out == "D") & ~p.cloud_mask.isin([0, 1])).sum())
    (OUT / "r2_calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
