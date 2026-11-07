# %%
"""Read-only v1/v2/v3 comparison for the single development night."""

from datetime import date
from pathlib import Path
import json

import numpy as np
import pandas as pd

import pipeline as p

HERE = Path(__file__).resolve().parent
NIGHT = "2026-10-29"
TAG = f"{NIGHT}_{NIGHT}"


def main(out_dir: Path = HERE / "outputs" / "dev_v3") -> dict:
    """Save comparison metrics; v1 has no isolated-cloud neighbour flags."""
    sites = pd.read_csv(p.SITES_CSV)
    result = {}
    for version, folder, tag in (
        ("v1", "v1", "2026-10-06_2026-11-04"),
        ("v2", "dev_v2", TAG),
        ("v3", "dev_v3", TAG),
    ):
        over = pd.read_csv(HERE / "outputs" / folder / f"overpasses_{tag}.csv")
        over = over.merge(sites[["site_id", "lon"]], on="site_id")
        local = pd.to_datetime(over.time_utc) + pd.to_timedelta(over.lon / 15, "h")
        over = over[
            ((local - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d") == NIGHT)
            & (over.sza >= p.SZA_CLEAR)
        ].drop(columns="lon")
        if "cloud_clear_neighbors" not in over:
            over["cloud_clear_neighbors"] = 0
        over = p.dedupe_overpasses(over)
        nights = p.site_nights(
            over, sites, date.fromisoformat(NIGHT), date.fromisoformat(NIGHT)
        )
        detected = over[over.detected]
        combos = detected.bands.fillna("").map(lambda s: "+".join(sorted(s.split("+"))))
        swir = detected.bands.fillna("").map(
            lambda s: not bool({"M12", "M13"} & set(s.split("+")))
        )
        result[version] = dict(
            strict_flaring=int((nights.n_detected > 0).sum()),
            detclear_flaring=int(over.loc[over.detected, "site_id"].nunique()),
            raw_detections=int(len(detected)),
            strict_detections=int(nights.n_detected.sum()),
            strict_rh_mw=float(nights.rh_mw.sum()),
            detclear_rh_mw=float(nights.rh_mw_detclear.sum()),
            band_shares=combos.value_counts(normalize=True).round(4).to_dict(),
            fit_flags=detected.fit_flag.value_counts().to_dict()
            if "fit_flag" in detected
            else {"unrecorded": len(detected)},
            swir_only_share=float(swir.mean()),
            temperature_p5_p50_p95=np.percentile(detected.temp_k, [5, 50, 95])
            .round(1)
            .tolist(),
            strict_note="cloud 0/1 only; neighbour flags unavailable"
            if version == "v1"
            else "strict including isolated-cloud rule",
        )
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# %%
if __name__ == "__main__":
    main()
