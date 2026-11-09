"""Night-level heat bounds for dark spells; nightly heat at 4497, the
4582/4583 complex and 4500 around the October stop; 4497 x complex association."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, "outputs/flareout_research/scripts_opus")
from common import passes  # noqa: E402

# pooled same-night re-detection probability by RH bin (s11): lower bin edges, p
CURVE = [(0.0, 0.190), (0.2, 0.341), (0.3, 0.532), (0.5, 0.712), (1.0, 0.792)]
SPELLS = {4582: ("2026-10-21", "2026-11-03"), 927: ("2026-10-30", "2026-11-04"),
          3662: ("2026-09-19", "2026-11-04"), 3644: ("2026-09-16", "2026-11-04"),
          6011: ("2026-09-16", "2026-11-04"), 3669: ("2026-10-23", "2026-11-04")}


def main():
    p = passes()
    p["c01"] = p.cloud_mask.isin([0, 1])
    for sid, (a, b) in SPELLS.items():
        x = p[(p.site_id == sid) & (p.night >= a) & (p.night <= b)]
        q = x[x.c01].groupby("night").size()
        qn, n1 = int((q >= 2).sum()), int((q >= 1).sum())
        det = int(x.detected.sum())
        out = []
        for lo, pr in CURVE:
            out.append(f">={lo}: {(1 - pr) ** n1:.1e}")
        # smallest bin excluded at 95% using one pass per night with >=1 clear pass
        ex = [lo for lo, pr in CURVE if (1 - pr) ** n1 <= 0.05]
        print(f"{sid} {a}..{b}: detections {det}; nights with >=1 clear pass {n1}, Q nights {qn}; "
              f"P(no detection | flame like bin), one pass per night: {'; '.join(out)}; "
              f"excluded from {ex[0] if ex else 'none'} MW up")
    d = p[p.detected & p.c01 & p.site_id.isin([4497, 4500, 4582, 4583])].copy()
    d["grp"] = d.site_id.map({4497: "4497", 4500: "4500", 4582: "cx", 4583: "cx"})
    t = d[d.night >= "2026-10-10"].groupby(["night", "grp"]).rh_mw.median().unstack()
    print(t.round(1).to_string())
    ct = np.array([[11, 25], [4, 17]])
    print("4497 x complex lit, 09-08..11-04, Fisher exact:", fisher_exact(ct))


if __name__ == "__main__":
    main()
