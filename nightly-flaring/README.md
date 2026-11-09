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

**Viewer:** https://skytruth.github.io/offshore-methane/ (30-night v5 run).
**How it works:** [docs/explainer.md](docs/explainer.md), also the viewer's
"How it works" tab.

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
| `backfill.py` | Detached backfill runner: warms the CMR cache, runs the pipeline with retries, validates; writes `status.json` (D50). |
| `gates_30n.py` | Evaluates the backfill gates (reviews.md, review 4) for a run folder against the v3 30-night files; writes `gates.json` and per-gate lists. |
| `build_viewer.py` | Joins every run segment under `outputs/v5_*` into one night series and builds the website in `outputs/site/` (page with data, basemap and explainer; per-site pass files; image chips); `publish_pages` deploys it to GitHub Pages (D51, D52). |
| `flareouts.py` | Source-centred High / Mid / Watch thermal-departure evidence cards; recalibrates from combined nights and copies on each build (D58). |
| `emit.py` | Public EMIT overpasses, browse links and optional Earth Engine CH4-index / true-colour chips cached under `outputs/emit_cache/` (D59). |
| `darknights.py` | Clear dark runs with per-night evidence, compact EMIT scene references and a persistent render queue (D62). |
| `update.py` | Processes new local nights, validates, rebuilds and refreshes EMIT; publishes only with `publish=True` (D60). |
| `docs/flareout_research/` | Research brief and six memos behind the flare-out method. |
| `viewer_template.html` | Page template for the viewer (d3 + topojson-client from CDN). |
| `tests/` | Unit tests for `nightfire.py` and the nightly aggregation. |
| `docs/` | `explainer.md` and its figures (`explainer/`: charts from `make_figures.py`, real VIIRS image chips from `make_chips.py`), `reviews.md` (council findings), `decisions.md` (local choices with evidence). |
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
.venv/bin/python build_viewer.py         # -> outputs/site/ (serve it: python -m http.server -d outputs/site)
.venv/bin/python -m pytest -q tests
```

The website maps every site for one night or the mean over a range of nights:
flaring sites glow on an amber scale sized by radiant heat. A timeline shows
the radiant heat seen each night over a band giving the share of sites in
clear sky; click a night, drag to average a range, use Last 7 / 30 / 90 / All,
or step with ← and →. Play animates the range; a radiant-heat filter keeps
the flares at or above a chosen percentile of mean radiant heat over the
selected nights, and an opened site always stays on the map. The flare list orders sites by
strength, consistency (share of clear nights with a flare, by the lower end of
its 95% interval) or clear views (clear passes), each row with a
one-column-per-night chart. Opening a site shows every night with a movable 14-night
window over its satellite passes, the matched EOG MYC25Q4 2025 flare, GFW
structure IDs and a satellite-view link. The side panel also totals by region
and compares the record with MYC25Q4. `build_viewer.py` takes every
`outputs/v5_*` segment (D52); pass files load over HTTP, so preview the folder
through a local server.

Scripts use `# %%` cells and a parameterised `main()`, per `AGENTS.md`. Dates
are local evening dates; night N uses overpasses on UTC dates N and N+1, which
must be within NOAA's ~91-day bucket retention.

## Monthly update and flare-out review

From `nightly-flaring/`, run:

```bash
.venv/bin/python -c "import update; update.main()"
```

This processes the next unprocessed night through today minus three days,
using `backfill.main` for CMR warming, processing and validation, then rebuilds
`outputs/site/`. It resumes interrupted update folders and checks their
immutable expected-job manifest. Default missing tolerance is zero; after retries,
`update.main(accept_missing=True)` explicitly records the missing granules and
marks affected site-nights not observed, allowing future months to proceed.
Silently vanished inputs cannot certify completeness. Before resuming,
at 75 days it warns about NOAA's roughly 91-day retention;
expired input requires archive recovery. Even when no new nights are due,
building refreshes EMIT imagery that Earth Engine may have ingested meanwhile.
Every monthly build also refreshes the compact `outputs/site/darknights.json`,
fetched when **Single dark nights** opens in Flare-outs (`#flare-outs/dark-nights`).
It lists one row per dark run: dates, clear dark count, qualifying-night count,
and the best night's rate (lit/clear), recent heat (n) and clear-pass count.
Each night needs >half of ≥10 clear nights lit in its ±14-night window,
excluding itself and nights with only a neighbour's detections, plus an earlier
lit clear night before its run inside that window. Rate-lit means a detection
within 1 km or nearest this source unit; neighbour-only nights are excluded
from both counts and their number is shown. Darkness still requires no
source-copy detection within 1.5 km. D58 card rules are unchanged. The default order combines the Wilson lower bound of that share
with the heat class's clear-pass detection term; detail shows both with n.
Passes on one night are not independent: this is a review order, not a probability.
Shared run details appear once and each qualifying night's evidence is stacked.
The strip marks lit nights without heat separately from unknown, caps high heat
with marked triangles, and labels the dark run. Short UTC pass times and normal
header wrapping fit phones; overlapping dark sources within 1.5 km are linked.
CMR keeps final monthly caches and batches all non-final months into one query
per product per source (two per rebuild, apart from pagination). Comparison
scenes exclude all listed runs' two-night windows and need measured EMIT data
at the source. Both modes check the index mask at the source and valid pixels
within 300 m; legacy chips migrate offline, with missing files kept unknown.
No-data tiles are final and keep browse links. Same-orbit tiles within 30 seconds
count once, using the tile that covers the source and noting adjacent tiles.
Only covered scenes count as with-chips. The current recipe (`hitl-v7`) shows
missing pixels in grey in both chips, inside and outside the scene, distinct
from the dark blue ≤1σ colour; true colour also masks EMIT edge fill (−9999 or
all-zero bands). It refreshes older covered chips; until then an older chip is
labelled as such. A dark night is not listed when another source within 1.5 km
had its own detections that night. The bounded render queue reserves new-scene
slots, backs off failures and keeps past-cutoff scenes pending even at zero
budget. Units, trimmed column heat histories and scene records appear once;
full pass histories stay private. See [D62](docs/decisions.md#d62-single-dark-nights-and-emit-comparison-scenes-v5)
for thresholds, cache rules and limits.
Publication is opt-in: `update.main(publish=True)` calls `publish_pages` only
after the build. Do not use it for a local preview.

The third tab, **Flare-outs** (`#flare-outs`), has a ranked list, locator map
and selected evidence detail, ordered by tier.
Cards include source heat, clear-dark nights, neighbours, pass lists and
EMIT review material. `outputs/site/flareouts.json` preserves the complete
payload and build calibration. The [Potential flare-outs method](docs/explainer.md#potential-flare-outs)
explains the tiers and limitations; [D58](docs/decisions.md#d58-source-centred-flare-out-evidence-cards-and-tiers-v5)
states the exact rolling rules and scan-family arithmetic; the screen cannot establish
extinction, venting or the cause of a stop. EMIT needs no auth for CMR/browse.
Earth Engine uses local credentials with project `cerulean-338116`; missing
credentials/network leaves an unavailable status and does not stop the build.
Chips follow `hitl_app._emit_truecolor`: L2A reflectance with validated L1B
radiance fallback, a shared stretch and gamma 1.8. Recent scenes beyond Earth
Engine's measured collection cutoff show that date and are retried; if it cannot
be measured, availability remains unknown. A listing becomes final only 30 days after its
window ends. Cached review chips survive offline rebuilds. No Earthdata token is needed or stored.

## Pipeline (v5)

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
   The final set requires two of M10/M11/M12/M13 and at least one of M10/M11.
   Use noise-weighted fits with a 10% model-error term. Evaluate each allowed
   leave-one-out trial with all guards, ranking guard-passing trials first,
   then reduced χ², retaining at least three bands during elimination. Accept 1,200–3,500 K and ESF in
   (0,1]; two-band fits require 1,200–2,500 K. Reject failed convergence or
   remaining high χ²; pin rejected fits at 1,810 K from usable M10, else leave
   unfitted. `fit_flag` is `ok`, `eliminated:<bands>`, `pinned:<reason>` or
   `unfitted`; pinned copies retain `t_fit_rejected`. Split a secondary M10
   maximum only when the connecting saddle is ≤0.5 times that maximum, then
   partition by nearest retained peak. A cluster needs a fitted M10 peak.
5. **Per overpass**: peak-anchored RH75% sums the peak and its eight neighbours
   within its partition and scan when fitted RH is 0.75–1.5 times peak RH.
   en2025 uses an RHmax anchor; our M10-peak anchor, stencil and cap are local
   choices. Pair adjacent-scan peaks with |Δcol|≤1, distance <1.5 times the
   smaller sqrt(footprint area), and RH75% ratio <3, closest pairs first.
   Each group has at most one peak per scan; consecutive-scan chains are allowed.
   Group RH is mean copy RH75%; assignment uses the RH-weighted spherical centroid
   and nearest site within 2 km. Distinct groups at a site are summed in `rh_mw`;
   `temp_k`, `area_m2`, bands and fit describe the strongest group's strongest copy.
   `n_sources` counts groups; `n_clusters` counts pre-merge confirmed copies.
   `peak_*` fields locate that strongest copy; `row`/`col` and cloud flags retain
   site-pixel semantics. `n_rh75` counts selected pixels from the strongest copy
   of each group. Cap, split and merged-copy counts support audits.
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

Per-granule caches are versioned as `overpasses_v5`; `out_dir` contains both
that cache and the final CSVs. Development runs use `outputs/dev_v5`, preserving
v1/v2/v3 outputs and caches.

Each cache entry needs both `<satellite>/<gid>.csv` and `<gid>_copies.csv`.
`copies_<tag>.csv` exports every confirmed copy, including unassigned groups:
peak position, scan/component, footprint, M10 prominence, RH75% and peak RH,
fit/bands/rejected temperature, peak cloud mask and clear neighbours, group ID,
size, mean/max RH, centroid, assigned site and distance. Accepted pair edge
fields (`pair_distance_px`, `pair_dcol`, `pair_drow`, `pair_rh_ratio`) are
semicolon-separated for chain endpoints with two edges; group IDs are scoped
by satellite and granule. Δrow is recorded without a cutoff.

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
| v5 | 2026-10-06 – 11-04 (30) | 19,443 | about 167 GB | about 105 min (3 attempts, D47) | 3,976 clear + 1,607 lit under cloud | 492 clear (548 incl. cloud) |


Dev night 2026-10-29, same local-night/SZA/dedupe scope (v3 = final v3):

| Measure | v3 (final) | v4 | v5 |
| --- | --- | --- | --- |
| Detected overpasses | 369.00 | 376.00 | 372.00 |
| Detected RH, MW | 1796.77 | 1823.48 | 1766.11 |
| Sum strict-clear site means, MW | 450.33 | 453.50 | 448.97 |
| Strict-clear detections | 257.00 | 255.00 | 256.00 |

V5 pinned share is 3.5% (13/372); split-cloud share is 80% (8/10 detection
records); multi-source edge share is 0% (0/4). MYC-centred strict frequency is
46.1% (263/571 passes), median 0.400 (n=188); detection-clears sensitivity is
53.5% (354/662), median 0.667 (n=229). Source locations use the copy table.
All 1,343/1,343 granules and 21/21 requested spot cases completed; 67 tests pass.

The 30-night v5 run is in `outputs/v5_30n/`; `gates_30n.py` writes its gate
results to `gates.json` and `gates/`. Against v3 over the same local nights:
detections 11,527 vs 11,695 (−1.4%), detected RH 60,146 vs 75,085 MW (−19.9%),
sum of site means 759 vs 944 MW (8.80 vs 10.95 BCM/yr). 81% of the RH change
corrects v3 defects (forbidden fits, RH75% summing remote pixels, one flare
counted at two sites); the mean of bow-tie copies accounts for −1,981 MW.
Council review 5 gave `BACKFILL: GO` from both reviewers.

[The v5 development evidence](docs/decisions.md#v5-development-evidence-2026-11-08)
contains the full case table, pair histogram, fit reasons, changed tests and risks.
Artifacts are in outputs/dev_v5: report.md, comparison.json, case_records.csv,
case_groups_v4.csv, case_groups.csv, pair_candidates.csv and copies_<tag>.csv.
Historical v4 evidence remains in decisions.md and outputs/dev_v4.

SNPP has no data in NOAA's bucket after 2026-11-02 15:32 UTC (no night passes
over sites after 12:51 UTC). v1 outputs are kept in `outputs/v1/`.

GFW sites with a detection on ≥1 clear night, by EOG MYC25Q4 2025 persistence
regime. v1 matched every GFW site within 1 km of a MYC site (many-to-one); v3
matches one-to-one within 2 km and uses the strict cloud rule, so the columns
are not strictly comparable.

| Regime | v1, 7 nights | v1, 30 nights | v3, 30 nights | v5, 30 nights |
| --- | --- | --- | --- | --- |
| persistent | 71% (235/331) | 85% (283/332) | 79% (283/356) | 79% (282/356) |
| intermittent | 31% (54/174) | 50% (88/175) | 42% (71/171) | 41% (70/171) |
| sporadic | 11% (8/72) | 18% (13/72) | 16% (11/67) | 16% (11/67) |
| undetected | 2.3% (3/130) | 3.8% (5/131) | 2.5% (3/118) | 2.5% (3/118) |

Annualised volume vs MYC25Q4 2025 at sites positive in both: v1 log-log r =
0.42 (n = 297, 7 nights) and 0.52 (n = 384, 30 nights), median ratio 0.83 (mean
of nightly means); v3 r = 0.51 (n = 365), median ratio 0.66 (pooled clear-sky
mean); v5 r = 0.51 (n = 363), median ratio 0.63. Flaring sites with no MYC
flare nearby: v1 369 (1 km), v3 136 (2 km), v5 126 (2 km).

## Reviews and decisions

- [docs/reviews.md](docs/reviews.md): council reviews. Review 1 covers the
  7-night v1 run and the v1 → v3 comparison; review 2 (2026-11-08) covers the
  30-night v3 run, with 12 new bugs (B1–B12) and the agreed fix order;
  reviews 3 and 4 cover v4 and v5 on the development night; review 5 covers
  the 30-night v5 run against gates 1–11 and the backfill decision.
- [docs/decisions.md](docs/decisions.md): every local choice and departure from
  the papers, with the reason and evidence, and which version introduced it.

## Round 3 validation

`validate_myc25.main(tag='2026-10-29_2026-10-29',
out_dir=OUT_DIR / 'dev_v3')` writes `validation_myc25_<tag>.csv` and
`validation_summary_<tag>.json`. Matches are one-to-one, accepting shortest
pairs first within 2 km. Period means pool all deduplicated strict clear
overpasses, including clear zeros, rather than averaging nightly means.
The original own-site strict-clear frequency remains. The added
`det_freq_like_for_like` is the detection-clears sensitivity metric: only source
group centroids within 2 km of MYC count as detections, over satellite pass groups
with <10 s consecutive gaps at GFW sites within 2 km. `det_freq_myc_strict` uses
cloud 0/1 plus the isolated-cloud rule. Nearby sites share a pass. Historical
outputs without copies use peak positions where available; their group-centroid
frequencies cannot be reconstructed. Regime counts and regional totals use matched sites
with at least one clear night; regional MYC totals use those same sites.
The default reference year is 2025, the latest annual catalog, so a one-night
2026 comparison is a diagnostic rather than annual validation.

QF1 layout source: [JPSS VIIRS SDR Data Dictionary Part 6, revision L](https://www.nesdis.noaa.gov/s3/2024-01/474-00448-02-06_JPSS-VIIRS-SDR-DD-Part-6_L.pdf).
Bits 2–3 encode none/some/all saturation as 0/1/2. Revision L labels the
partial state unimplemented: QF1 cannot guarantee detection of all subpixel
saturation; weighted negative-residual elimination is also needed.
`verify_qf1.main()` downloads one M12 granule and records flag counts and
raw-radiance agreement under `outputs/dev_v3/`.
