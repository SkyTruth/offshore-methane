# %%
"""Read-only v1/v2/v3/v4/v5 comparison for the single development night.

``v3`` is the final v3 code as run for 30 nights; ``v3_dev`` is the round-3
development output, which predates the 10% model-error term.
"""

from datetime import date
from pathlib import Path
import json

import numpy as np
import pandas as pd

import pipeline as p

HERE = Path(__file__).resolve().parent
NIGHT = "2026-10-29"
TAG = f"{NIGHT}_{NIGHT}"


def main(out_dir: Path = HERE / "outputs" / "dev_v5") -> dict:
    """Save comparison metrics; v1 has no isolated-cloud neighbour flags."""
    sites = pd.read_csv(p.SITES_CSV)
    result = {}
    for version, folder, tag in (
        ("v1", "v1", "2026-10-06_2026-11-04"),
        ("v2", "dev_v2", TAG),
        ("v3_dev", "dev_v3", TAG),  # round-3 code, before the 10% model error
        ("v3", ".", "2026-10-06_2026-11-04"),  # final v3 code (30-night run)
        ("v4", "dev_v4", TAG),
        ("v5", "dev_v5", TAG),
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
        raw = over.copy()
        legacy = over.copy()
        legacy["_detected"] = legacy.detected
        legacy["detected"] = False
        legacy = p.dedupe_overpasses(legacy)
        legacy["detected"] = legacy.pop("_detected")
        over = p.dedupe_overpasses(over) if version in ("v4", "v5") else legacy
        nights = p.site_nights(
            over, sites, date.fromisoformat(NIGHT), date.fromisoformat(NIGHT)
        )
        detected = over[over.detected]
        combos = detected.bands.fillna("").map(lambda s: "+".join(sorted(s.split("+"))))
        swir = detected.bands.fillna("").map(
            lambda s: not bool({"M12", "M13"} & set(s.split("+")))
        )
        two = detected[
            detected.bands.fillna("").map(
                lambda x: len(x.split("+")) == 2 and "T1810" not in x
            )
        ]
        metrics = dict(
            two_band_fits=len(two),
            two_band_temperature_range=[
                float(two.temp_k.min()),
                float(two.temp_k.max()),
            ],
            fits_below_1200=int((detected.temp_k < 1200).sum()),
            total_detected_rh=float(detected.rh_mw.sum()),
            median_detected_rh=float(detected.rh_mw.median()),
            flaring_sites_including_cloud=int(
                nights.status.isin(["flaring", "flaring_cloud"]).sum()
            ),
            duplicate_rows_removed=len(raw) - len(over),
            dedupe_detection_gains=int(over.detected.sum() - legacy.detected.sum()),
        )
        peak_rh = (
            detected.rh_peak_mw
            if version in ("v4", "v5")
            else (5.670374419e-8 * detected.temp_k**4 * detected.area_m2 / 1e6)
        )
        ratios = (detected.rh_mw / peak_rh).replace([np.inf, -np.inf], np.nan).dropna()
        metrics.update(
            rh75_over_peak_p5_p50_p95=ratios.quantile([0.05, 0.5, 0.95]).tolist(),
            rh75_over_peak_n=len(ratios),
            peak_rh_note="recorded"
            if version in ("v4", "v5")
            else "reconstructed from rounded T/source area",
        )
        if version in ("v4", "v5"):
            metrics.update(
                rh75_over_peak_p5_p50_p95=ratios.quantile([0.05, 0.5, 0.95]).tolist(),
                rh75_over_peak_n=len(ratios),
                cap_excluded_pixels=int(detected.n_rh_cap_excluded.sum()),
                fitted_pixels=int(detected.n_fitted.sum()),
                components_split=sum(
                    len(
                        {
                            part
                            for value in group.split_component_ids.fillna("").astype(
                                str
                            )
                            for part in value.split(",")
                            if part
                        }
                    )
                    for _, group in detected.groupby(["satellite", "granule"])
                ),
                copies_merged=int(detected.n_copies_merged.sum()),
                multisource_records=int((detected.n_sources > 1).sum()),
                multisource_sites=int(
                    detected.loc[detected.n_sources > 1, "site_id"].nunique()
                ),
            )
        import validate_myc25 as validation

        myc = validation.load_myc()
        copies = (
            pd.read_csv(HERE / "outputs/dev_v5" / f"copies_{TAG}.csv")
            if version == "v5"
            else None
        )
        matched = validation.match_sites(
            nights, myc, overpasses=raw if version == "v5" else over, copies=copies
        )
        summary = validation.validation_summary(matched)
        metrics["det_freq_myc_strict"] = summary["det_freq_myc_strict"]
        metrics["det_freq"] = summary["det_freq"]
        metrics["det_freq_like_for_like"] = summary["det_freq_like_for_like"]
        result[version] = dict(
            **metrics,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# %%
if __name__ == "__main__":
    main()
