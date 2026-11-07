# Nightly flaring

Side project: per-overpass VIIRS flare detection at offshore oil infrastructure,
following the published VIIRS Nightfire (VNF) method from the Earth Observation
Group (EOG, Colorado School of Mines). The target output is one row per site per
night overpass: clear or cloudy, detected or not, temperature, source area,
radiant heat, flared-volume rate.

Since 2025-01-10, nightly VNF data requires the VIIRS Nightfire Data Use License.
The license is free for academics for one year, and a paid annual subscription
for nonprofits and commercial users (price unpublished; contact eog@mines.edu).
Annual data (MYC25Q4, 2012-2025, 27,229 upstream sites) remains free on Zenodo
under CC BY-NC 4.0: <https://doi.org/10.5281/zenodo.23091822>.

## Contents

| File | Purpose |
| --- | --- |
| `fetch_gfw_oil.py` | Pulls the GFW fixed-infrastructure layer, keeps active `label == "oil"` structures, merges structures within 750 m into sites. |
| `granule_coverage.py` | Groups sites by the VIIRS night granules that cover them on one date, from NASA CMR footprints. |
| `pipeline.py` | Date-range run: selects granules, processes them in parallel, writes one row per site per night. |
| `granule.py` | Processes one granule: site pixels, M10 detection, M11-M13 confirmation, Planck fit, cloud mask. |
| `nightfire.py` | Physics: Planck function, zone thresholds, Planck fit, pixel area, radiant heat, solar zenith. |
| `nodd.py` | NOAA bucket listing, granule index, HTTP byte-range file object for h5py. |
| `validate_myc25.py` | Matches a run to EOG's MYC25Q4 annual sites and reports detection rates and volume agreement. |
| `tests/` | Unit tests for `nightfire.py` and the nightly aggregation. |
| `outputs/` | Generated CSVs and the per-granule cache (git-ignored). |

## Running

Dependencies are the `methane` env plus `requirements.txt` (h5py,
mapbox-vector-tile, requests, scikit-learn). A local venv works:

```bash
cd nightly-flaring
/path/to/conda/env/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

Set `INFRA_API_TOKEN` (GFW gateway token; production keeps it in GCP Secret
Manager as `production-gfw-infra-api-key`) in the shell or the repo-root `.env`.

```bash
.venv/bin/python fetch_gfw_oil.py        # -> outputs/gfw_oil_{points,sites}.csv
.venv/bin/python granule_coverage.py     # -> outputs/granule_sites_<date>.csv
.venv/bin/python -c "import pipeline; pipeline.main(start='2026-10-29', end='2026-11-04', workers=12)"
.venv/bin/python validate_myc25.py
.venv/bin/python -m pytest -q tests
```

Scripts use `# %%` cells and a parameterised `main()`, per `AGENTS.md`. Dates
are local evening dates; night N uses overpasses on UTC dates N and N+1, which
must be within NOAA's ~91-day bucket retention.

## Pipeline (first pass)

1. **Granule selection**: NASA CMR footprints of the 6-minute geolocation
   granules, split into four slices, select the NOAA 86 s granules (±20 s) and
   the sites each one can see.
2. **Site pixel**: nearest valid pixel in the terrain-corrected geolocation
   (`GMTCO`) within 1.2 km; sites on the granule border are left to the
   neighbouring granule.
3. **Detection (stage 1, every granule)**: M10 radiance above the
   aggregation-zone mean + 4σ (one 4σ clip of bright pixels stands in for the
   2013 DN > 100 screen), solar zenith ≥ 95°, within ±2 pixels of a site.
4. **Confirmation and fit (stage 2, only granules with a candidate)**: M11
   (zone mean + 4σ), M12/M13 (10×10 local background + 3σ, background
   subtracted). Pixels with M10 plus at least one other band get a
   single-curve Planck fit for temperature and emission scaling factor;
   saturated bands are dropped. M10-only pixels are counted
   (`n_hot_m10`) but not treated as detections.
5. **Per overpass**: hot pixels are assigned to the nearest site within 2 km.
   Radiant heat sums the pixels in the brightest pixel's scan with RH ≥ 0.75 ×
   max (RH75%, Zhizhin 2025). The cloud mask (NOAA enterprise `JRR-CloudMask`)
   is read at the site pixel.
6. **Per night** (`site_nights_<start>_<end>.csv`): overpasses with solar zenith
   ≥ 97° are pooled across SNPP, NOAA-20 and NOAA-21. Clear = detected or cloud
   mask clear/probably clear. `rh_mw` = Σ RH over clear overpasses / clear
   overpasses (clear non-detections are 0); `bcm_per_year` = 0.0116 × `rh_mw`.
   `status` is `flaring`, `clear_no_flare`, `cloudy` or `not_observed`.

Simplifications versus EOG: Planck at band centres (no spectral response),
single-curve fit with background subtraction instead of the dual-curve fit, no
M7/M8/DNB, no atmospheric correction, no top-hat glow filter, no isolated-cloud
clearing, and no cross-satellite calibration.

## Points of interest

GFW `public-fixed-infrastructure-filtered:latest`, fetched as one z0 vector tile
(same call as `cerulean_cloud/cloud_function_asa/utils/analyzer.py`). On
2026-11-07: 59,863 structures, 32,951 labelled oil, 11,342 oil structures active
through the latest month (2026-06-01), 7,637 sites after merging at 750 m (one
M-band pixel at nadir).

## Coverage and data volume (2026-10-25)

| Satellite | Night/both 6-min granules | With ≥1 site | ~86 s slices with ≥1 site |
| --- | --- | --- | --- |
| SNPP | 125 | 69 | 154 / 500 |
| NOAA-20 | 126 | 68 | 156 / 504 |
| NOAA-21 | 122 | 64 | 139 / 488 |

Inputs come from NOAA's public buckets `noaa-nesdis-{snpp,n20,n21}-pds`
(free, anonymous, ~91-day rolling retention; the oldest file on 2026-11-07 was
from 2026-08-08). Storage layout:

- SDR band files (`VIIRS-Mxx-SDR/`, `SVMxx_*.h5`): each 768×3200 array is a
  single deflate chunk, so a band is downloaded whole per granule.
- Geolocation (`VIIRS-MOD-GEO-TC/`, `GMTCO_*.h5`, ~27 MB): latitude and
  longitude are separate ~3.3 MB chunks, readable by byte range without the
  rest of the file.
- Cloud mask (`VIIRS-JRR-CloudMask/`, ~38 MB): chunked 256×200, so windows
  around sites can be read alone.

Mean file sizes per granule (day and night files, NOAA-20, 2026-11-01), in MB:
M7 8.3, M8 4.0, M10 3.8, M11 3.8, M12 5.6, M13 10.8, M14-M16 ~5.9 each.
M7, M8, M10-M13 plus lat/lon total ~43 MB per granule; adding M14-M16 gives
~60 MB. At ~150 granules per satellite per night that is 6.7-9.3 GB per
satellite, 20-28 GB per night for three satellites. EOG reports 150 GB of night
data per instrument per night for the global run (Elvidge 2023).

Backfill beyond 91 days needs NOAA CLASS SDR orders or NASA LAADS L1B
(`VNP02MOD`/`VJ102MOD`), whose calibration differs from NOAA's IDPS SDRs.

## Published method

All open access (MDPI); DOI prefix 10.3390.

| Reference | Specifies |
| --- | --- |
| Elvidge et al. 2013, *Remote Sens.* 5:4423, `rs5094423` | M10 detection, M12/M13 local background, Planck fit, footprint equations, radiant heat |
| Elvidge et al. 2019, *Remote Sens.* 11:395, `rs11040395` | M11, M12-M13 scattergram detector, dual Planck fit, band elimination, detection limits |
| Zhizhin et al. 2023, *Remote Sens.* 15:1189, `rs15051189` | M12-M13 detector details (not yet reviewed) |
| Elvidge et al. 2021, *Fire* 4:83, `fire4040083` | v4: flaming subtractive fit, atmospheric correction, saturation |
| Elvidge et al. 2023, *Remote Sens.* 15:4760, `rs15194760` | Operational nightly pipeline, cloud handling, site catalog |
| Zhizhin et al. 2021, *Remote Sens.* 13:3078, `rs13163078` | Site detection, cloud-free normalisation, Cedigaz calibration |
| Elvidge et al. 2016, *Energies* 9:14, `en9010014` | First volume calibration (RH′ = σT⁴S^0.7 vs Cedigaz) |
| Zhizhin et al. 2025, *Energies* 18:4765, `en18174765` | John Zink controlled-flare calibration (36 tests) |
| Zhizhin et al. 2026, *Remote Sens.* 18:314, `rs18020314` | MYC25 super-resolution site catalog |
| Zenodo MYC25Q4 README | Clear-overpass rule, 0.0116 BCM/(yr·MW), persistence regimes |

### Steps

1. Night only: drop pixels with solar zenith < 95°.
2. M7/M8/M10/M11: DN > mean + 4σ, with statistics per granule and per
   aggregation zone, excluding DN > 100 from the background (2013 §2.1; 2019).
3. M12/M13: mean + 3σ in a 10×10 window around M10 hot pixels, expanded to
   100×100 below 50 background pixels (2013); later an M12-vs-M13 scattergram
   with a convex-hull background (2019; Zhizhin 2023).
4. Filters: confirmation in ≥2 bands (particle hits); detections from adjacent
   scans ≥ half a pixel apart (bow-tie); full-pixel saturation per 2019 Table 1.
5. Planck fit: Nelder-Mead on temperature and emission scaling factor (ESF),
   start 1000 K / 1.0. Dual curve (emitter + background) when MWIR detects.
   If SSR > 2, drop the band with the largest negative residual and refit.
   M10 + DNB only: T fixed at 1810 K.
6. Source area = ESF × pixel footprint (2013 Eqs. 4-7); radiant heat (MW) =
   ESF·σT⁴ × footprint.
7. Per-overpass value: RH75% = sum of RH over cluster pixels above
   0.75·RHmax within one 16-line scan (2025).
8. Clear overpass: solar zenith > 97°, site inside a valid (not bow-tie-deleted)
   M-band pixel, cloud mask confidently or probably clear (Zenodo README).
9. Annual: mean RH = Σ clear-sky RH / clear overpasses, with clear
   non-detections as 0; BCM/yr = 0.0116 × mean RH in MW.

### Not published

- Fit objective, weighting, and spectral-response handling; SSR units behind the
  threshold of 2.
- Scattergram density cut, top-hat kernel size, local-maximum neighbourhood,
  isolated-cloud clearing rule.
- Atmospheric-correction settings (v4 only).
- Final John Zink slopes for v3/v4 appear only in figures; the 2025 text gives
  RH75% = 80.846 MW per BCM/yr (0.0124), the Zenodo release uses 0.0116.
- SNPP / NOAA-20 / NOAA-21 cross-calibration; Zenodo states satellite-specific
  volumes must not be added.
- Per-detection instantaneous volume formula; no code or ATBD released.

Validation target: MYC25Q4 annual site values for sites present in both
catalogs.

### Reference point

VIIRS 375 m active fire (free via NASA FIRMS) is a baseline. In the 2013
comparison, Nightfire found 94% (1319/1400) of the combined detections and
VIIRS active fire 59% (822/1400).
