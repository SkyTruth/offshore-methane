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
| `build_viewer.py` | Builds `outputs/flaring_viewer.html`, a self-contained map viewer (data and basemap inlined). |
| `viewer_template.html` | Page template for the viewer (d3 + topojson-client from CDN). |
| `tests/` | Unit tests for `nightfire.py` and the nightly aggregation. |
| `docs/` | `reviews.md` (council findings) and `decisions.md` (local choices with evidence). |
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
.venv/bin/python build_viewer.py         # -> outputs/flaring_viewer.html
.venv/bin/python -m pytest -q tests
```

The viewer maps every site for one night or the mean over a range of nights:
flaring sites glow on an amber scale sized by radiant heat, and hovering shows
a sparkline of the nightly values. A timeline strip shows the radiant heat seen
each night; click a night, or drag to average a range. Play animates the range,
easing each site's brightness between nights (cloudy nights hold the last clear
value at low opacity); a percentile slider hides sites below a chosen
radiant-heat percentile. `build_viewer.py` builds from the run with the latest
end date and longest span unless `start`/`end` are given. Clicking pins a per-overpass chart, the
matched EOG MYC25Q4 2025 flare, GFW structure IDs and a satellite-view link.
The side panel ranks the strongest flares, totals by region, and compares the
run with MYC25Q4.

Scripts use `# %%` cells and a parameterised `main()`, per `AGENTS.md`. Dates
are local evening dates; night N uses overpasses on UTC dates N and N+1, which
must be within NOAA's ~91-day bucket retention.

## Pipeline (v4)

1. **Granule selection**: NASA CMR footprints of the 6-minute geolocation
   granules, split into four slices, select the NOAA 86 s granules (±20 s) and
   the sites each one can see.
2. **Site pixel**: nearest finite-M10, valid geolocation pixel within 1.2 km;
   sites on the outermost granule border are left to the neighbouring granule.
3. **Detection (stage 1)**: M10 radiance above the aggregation-zone mean + 4σ,
   solar zenith ≥ 95°. Label 8-connected hot pixels separately in each 16-line
   scan. Select full clusters touching a clipped ±2-pixel site window. Reject a
   cluster if any M10 pixel in the peak's same-scan ±2 window is brighter.
4. **Confirmation and fit (stage 2)**: fit every selected cluster pixel using
   M7/M8/M10/M11 (zone mean + 4σ), and background-subtracted M12/M13 (local mean +
   3σ). Drop saturated bands using raw radiance, before background subtraction.
   QF1 bits 2–3 also exclude partially/fully saturated bands. Missing M7/M8
   files are optional. At least two unsaturated bands confirm a candidate;
   M10 or M11 must survive. Use noise-weighted fits with a 10% model-error
   term. Above reduced χ² 9, remove the band whose leave-one-out refit lowers
   χ² most, retaining at least three bands. Accept 1,200–3,500 K and ESF in
   (0,1]; two-band fits require 1,200–2,500 K. Reject failed convergence or
   remaining high χ²; pin rejected fits at 1,810 K from usable M10, else leave
   unfitted. `fit_flag` is `ok`, `eliminated:<bands>`, `pinned:<reason>` or
   `unfitted`. Split components around separate qualifying M10 maxima by
   nearest-peak pixel distance. A cluster needs a fitted M10 peak.
5. **Per overpass**: RH75% sums fitted pixels with RH between 0.75 and 1.5
   times the M10 peak RH. Merge adjacent-scan peaks closer than one pixel
   (smaller local sqrt(footprint area)), including chains; retain the strongest
   RH75% copy. Assign each source to the nearest site within 2 km and sum
   distinct sources. The strongest source supplies temperature, area and fit.
   `n_sources` counts distinct sources; `n_clusters` counts pre-merge copies.
   `rh_peak_mw` and `n_rh75` record peak RH and summed pixel count;
   `n_rh_cap_excluded`, `n_copies_merged`, `n_components_split` and
   `split_component_ids` support audits. `peak_row`, `peak_col`, `peak_lat`,
   `peak_lon`, `peak_dist_m` locate the strongest source; `row`/`col` remain
   the site pixel. Cloud flags and clear-neighbour counts use that site pixel.
6. **Per night**: deduplicate same-satellite/site records in runs of consecutive
   time gaps <10 s, preferring detections, then smallest `dist_m` (earliest on
   ties), then pool
   overpasses with solar zenith ≥ 97°. Clear = cloud mask 0/1 OR a detection
   surrounded by ≥6 clear/probably-clear neighbours. `rh_mw` averages over strict
   clear observations, including clear non-detections as zero; `n_detected`,
   `rh_max_mw` and `temp_k` also use strict clearing. Detections only
   under non-isolated cloud yield `flaring_cloud`, even with a clear zero.
   Precedence: `flaring` > `flaring_cloud` > `clear_no_flare` > `cloudy` >
   `not_observed`; `rh_mw` remains the strict-clear mean and can be zero.
   `n_clear_detclear`, `rh_mw_detclear`, `bcm_per_year_detclear` preserve the old
   detected-OR-cloud-0/1 rule on the same deduplicated observations. Volume rate
   is 0.0116 × mean RH.

Per-granule caches are versioned as `overpasses_v4`; `out_dir` contains both
that cache and the final CSVs. Development runs use `outputs/dev_v4`, preserving
v1/v2/v3 outputs and caches.

Simplifications versus EOG: Planck at band centres (no spectral response),
single-curve fit with background subtraction instead of the dual-curve fit, no
DNB, no atmospheric correction, and no cross-satellite calibration. The
local-maximum window and isolated-cloud neighbour threshold are explicit local
choices because their exact operational parameters are not published.

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

## Runs

| Pipeline | Nights | Granules | Downloaded | Wall time (12 workers) | Flaring site-nights | Flaring sites |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 2026-10-29 – 11-04 (7) | 4,551 | 32 GB | 21 min | 1,804 | 506 |
| v1 | 2026-10-06 – 11-04 (30) | 19,443 | 143 GB | 71 min for 23 new nights | 7,691 | 758 |
| v3 | 2026-10-06 – 11-04 (30) | 19,443 | 167 GB | 105 min | 4,060 clear + 1,634 lit under cloud | 504 clear (558 incl. cloud) |


Dev night 2026-10-29, same local-night/SZA/dedupe scope (v3 = final v3 code):

| Measure | v3 | v4 |
| --- | --- | --- |
| Detected overpasses | 369 | 376 |
| Total detected RH, MW | 1,796.8 | 1,823.5 |
| Strict-clear site RH means summed, MW | 450.3 | 453.5 |
| Fits below 1,200 K | 3/369 | 0/376 |
| Two-band fits above 2,500 K | present (max 3,202 K) | 0/48 |
| Flaring sites, strict / including cloud | 122 / 177 | 120 / 179 |
| MYC-centred frequency, median | 0.667 (n=229) | 0.667 (n=231) |

The large v3 failures (B1, B2) fall on other nights; the regression granules
show them: site 3802 3,127 → 124 MW, site 3804 1,776 → 71 MW.

Full counts, fit reasons, regression granules, defaults and open risks are in
[the v4 development evidence](docs/decisions.md#v4-development-evidence-2026-11-08).
Machine-readable results are in `outputs/dev_v4/comparison.json` and
`outputs/dev_v4/regressions.csv`. No 30-night v4 run has been performed.

SNPP has no data in NOAA's bucket after 2026-11-02 15:32 UTC (no night passes
over sites after 12:51 UTC). v1 outputs are kept in `outputs/v1/`.

GFW sites with a detection on ≥1 clear night, by EOG MYC25Q4 2025 persistence
regime. v1 matched every GFW site within 1 km of a MYC site (many-to-one); v3
matches one-to-one within 2 km and uses the strict cloud rule, so the columns
are not strictly comparable.

| Regime | v1, 7 nights | v1, 30 nights | v3, 30 nights |
| --- | --- | --- | --- |
| persistent | 71% (235/331) | 85% (283/332) | 79% (283/356) |
| intermittent | 31% (54/174) | 50% (88/175) | 42% (71/171) |
| sporadic | 11% (8/72) | 18% (13/72) | 16% (11/67) |
| undetected | 2.3% (3/130) | 3.8% (5/131) | 2.5% (3/118) |

Annualised volume vs MYC25Q4 2025 at sites positive in both: v1 log-log r =
0.42 (n = 297, 7 nights) and 0.52 (n = 384, 30 nights), median ratio 0.83 (mean
of nightly means); v3 r = 0.51 (n = 365), median ratio 0.66 (pooled clear-sky
mean). Flaring sites with no MYC flare nearby: v1 369 (1 km), v3 136 (2 km).

## Reviews and decisions

- [docs/reviews.md](docs/reviews.md): council reviews. Review 1 covers the
  7-night v1 run and the v1 → v3 comparison; review 2 (2026-11-08) covers the
  30-night v3 run, with 12 new bugs (B1–B12) and the agreed fix order.
- [docs/decisions.md](docs/decisions.md): every local choice and departure from
  the papers, with the reason and evidence, and which version introduced it.

## Round 3 validation

`validate_myc25.main(tag='2026-10-29_2026-10-29',
out_dir=OUT_DIR / 'dev_v3')` writes `validation_myc25_<tag>.csv` and
`validation_summary_<tag>.json`. Matches are one-to-one, accepting shortest
pairs first within 2 km. Period means pool all deduplicated strict clear
overpasses, including clear zeros, rather than averaging nightly means.
The original own-site strict-clear frequency remains. The added
`det_freq_like_for_like` compares detections at any GFW site within 2 km of a
matched MYC flare per strict-clear-or-detected satellite pass with MYC
n_det/n_clear; nearby sites share a pass. Regime counts and regional totals use matched sites
with at least one clear night; regional MYC totals use those same sites.
The default reference year is 2025, the latest annual catalog, so a one-night
2026 comparison is a diagnostic rather than annual validation.

QF1 layout source: [JPSS VIIRS SDR Data Dictionary Part 6, revision L](https://www.nesdis.noaa.gov/s3/2024-01/474-00448-02-06_JPSS-VIIRS-SDR-DD-Part-6_L.pdf).
Bits 2–3 encode none/some/all saturation as 0/1/2. Revision L labels the
partial state unimplemented: QF1 cannot guarantee detection of all subpixel
saturation; weighted negative-residual elimination is also needed.
`verify_qf1.main()` downloads one M12 granule and records flag counts and
raw-radiance agreement under `outputs/dev_v3/`.
