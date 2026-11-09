# Decisions

Local choices where the papers give no value, and departures from the papers.
Each entry names the version that introduced it. A change adds a new entry and
marks the old one `superseded by Dnn`; entries are not deleted. Review findings
(B1–B12, review-1 fixes 1–9) are in [reviews.md](reviews.md).

Format: **Choice**, **Alternatives**, **Why**, **Evidence**, **Status**.

## Inputs and coverage

### D01 Sites: GFW oil structures merged within 750 m (v1)
- **Choice:** GFW fixed infrastructure, `label == "oil"`, active (no end date or
  seen in the latest month); structures within 750 m merged by DBSCAN
  (haversine, `min_samples=1`). 7,637 sites.
- **Alternatives:** individual structures (11,342); flare-centred basins.
- **Why:** 750 m is about one M-band pixel at nadir; flares closer than that
  are not separable.
- **Evidence:** 3,858 of 7,637 sites have another site within 2 km, which
  drives the association problems in review 2.
- **Status:** active; basins are fix 4 of review 2.

### D02 Data: NOAA IDPS SDRs from the NODD buckets (v1)
- **Choice:** `noaa-nesdis-{snpp,n20,n21}-pds`, M7/M8/M10–M13 SDR, GMTCO
  geolocation, JRR-CloudMask, read by HTTP byte range.
- **Alternatives:** NASA LAADS L1B (VNP02MOD/VJ102MOD/VJ202MOD, archive from
  2012/2018/2023); NOAA CLASS orders.
- **Why:** same SDRs as EOG's operational run; free, anonymous, no egress
  charge.
- **Evidence:** retention probed 2026-11-08: no files before 2026; first full
  day 2026-08-09 (NOAA-20/21) and 2026-08-11 (SNPP).
- **Status:** active. Backfill beyond about 91 days needs another source.

### D03 Granule selection from NASA CMR footprints (v1)
- **Choice:** 6-minute VNP03MOD/VJ103MOD/VJ203MOD footprints split into 4
  along-track slices; a NOAA 86 s granule is processed when a slice with ≥1
  site overlaps it within ±20 s.
- **Alternatives:** read GMTCO for every night granule.
- **Why:** NOAA buckets carry no footprint metadata; about 150 of 500 night
  granules per satellite contain a site.
- **Status:** active.

### D04 Local night (v1)
- **Choice:** local solar time = UTC + lon/15 h; night = date of (local − 12 h).
  Night N uses UTC dates N and N+1.
- **Status:** active.

## Detection

### D05 Solar zenith limits (v1)
- **Choice:** detect at SZA ≥ 95° (rs2013); count a pass as a clear
  observation only at SZA ≥ 97° (MYC25Q4 README).
- **Status:** active.

### D06 Site pixel (v3, revised from v1)
- **Choice:** nearest pixel with valid geolocation and finite M10 radiance,
  within 1.2 km of the site; sites on the outermost granule border are left
  to the neighbouring granule.
- **Alternatives:** v1 used geolocation only.
- **Why:** bow-tie-deleted pixels have geolocation but fill radiance.
- **Evidence:** v1: 11% (21,205/189,764) of site-overpasses used fill pixels.
- **Status:** active.

### D07 Zone thresholds (v1)
- **Choice:** per granule and per aggregation zone (columns 0–640, 640–1008,
  1008–2192, 2192–2560, 2560–3200): mean + 4σ after one 4σ clip; the clipped
  σ is also the band noise in the fit.
- **Alternatives:** rs2013's DN > 100 background screen; pooling symmetric
  zones.
- **Why:** inputs are radiances; no per-band radiance equivalent of DN 100 is
  published.
- **Evidence:** review 2: zones estimated separately; effect unmeasured.
- **Status:** active.

### D08 MWIR local background (v1)
- **Choice:** M12/M13 mean and σ in a 10×10 window (100×100 if fewer than 50
  pixels), excluding M10-hot pixels; hot when above mean + 3σ (rs2013).
- **Alternatives:** M12–M13 scattergram detector (rs2019, rs2023b).
- **Evidence:** review 2: M12 glow pixels are not excluded (inferred effect:
  fewer MWIR confirmations at large flares).
- **Status:** active.

### D09 Trigger and search window (v1)
- **Choice:** candidates start from M10 hot pixels only; a cluster must touch
  a ±2-pixel window around the site pixel.
- **Alternatives:** M11 and M12/M13 triggers (rs2019, rs2023b).
- **Evidence:** review 2: 59 of 356 persistent MYC sites have no detection;
  their MYC median is 0.002 BCM/yr.
- **Status:** active; triggers are fix 6 of review 2.

### D10 Clusters and glow test (v3)
- **Choice:** 8-connected M10 hot pixels, labelled separately in each 16-line
  scan; peak = brightest M10 pixel; keep a cluster only if no M10 pixel within
  ±2 pixels in the same scan is brighter than the peak. ±2 is a local choice
  (rs2019 gives no neighbourhood size).
- **Evidence:** glow detections fell from 12.7% (501/3,935, v1) to 0.97%
  (116/11,924, v3). Review 2 B6: connected flares merge into one cluster.
- **Status:** superseded by D24 (v4).

### D11 Confirmation and saturation (v3)
- **Choice:** a pixel is confirmed by ≥ 2 unsaturated bands; a cluster counts
  only if its M10 peak is confirmed and fitted. Saturation from raw radiance
  against rs2019 Table 1 before background subtraction, plus QF1 bits 2–3.
- **Evidence:** QF1 partial saturation is labelled unimplemented in the SDR
  data dictionary rev. L; dev check flagged 2/2 near-saturation M12 pixels
  (1 granule).
- **Status:** active.

## Fit and radiant heat

### D12 Planck fit form (v3)
- **Choice:** single Planck curve at band centres, MWIR bands background-
  subtracted; Nelder–Mead on T and log ESF; residuals divided by band noise
  (zone σ or local σ, floor 1e-6) added in quadrature to a 10% relative model
  error.
- **Alternatives:** rs2019 dual-curve fit; spectral response functions.
- **Why (10%):** band-centre Planck, no atmospheric correction and calibration
  differences leave a few percent of model error; without the term, bright
  flares fail χ² on model error alone.
- **Evidence:** without it 100% of fits above 20 MW eliminated bands
  (2026-10-29); with it, elimination in the 12 brightest granules fell from 41%
  to 2%.
- **Status:** active.

### D13 Fit acceptance and elimination (v3)
- **Choice:** reduced χ² = χ²/max(n − 2, 1) ≤ 9; while above 9 with > 2 bands,
  drop the most negative normalised residual and refit; accept 600–3,500 K,
  ESF in (0, 1], converged; otherwise pin T = 1,810 K from M10, otherwise
  unfitted.
- **Evidence:** review 2 B1 (elimination can leave M7+M8 only: 8 fits, about
  0.99 BCM/yr), B3 (two-band χ² ≡ 0: 2,107 fits, 238 above 2,500 K), B4 (600 K
  floor: 76 detections below 1,200 K hold 9% of radiant heat).
- **Status:** superseded by D22 (v4).

### D14 Source area and radiant heat (v1)
- **Choice:** source area = ESF × pixel footprint, footprint from neighbouring
  pixel-centre spacing; RH = σT⁴ × source area, in MW. No view-angle
  correction.
- **Evidence:** review 2: 0.573 km² at nadir, 2.60 km² at swath edge; per-pixel
  RH 1.21× site median at swath edge vs 0.86× at nadir.
- **Status:** active.

### D15 RH75% and per-site value (v3)
- **Choice:** per cluster, sum fitted pixels with RH ≥ 0.75 × the cluster's
  maximum pixel RH; cluster assigned to the site nearest its M10 peak within
  2 km; per-site value is the strongest cluster (treats other clusters as scan
  copies).
- **Evidence:** review 2 B2 (threshold from the highest-RH pixel: site 3804
  reports 1,776 MW vs 71 MW at its peak), B5 (scan copies split across sites),
  B7 (963 detections with > 1 cluster).
- **Status:** superseded by D23 (v4).

## Clouds and nightly values

### D16 Clear-pass rule (v3, revised from v1)
- **Choice:** JRR-CloudMask at the site pixel; clear = confidently or probably
  clear; a detection under a cloud flag counts as clear only when ≥ 6 of its 8
  neighbours are clear (rs2021 clears small isolated clouds over flares but
  gives no size).
- **Alternatives:** v1 counted every detection as clear.
- **Evidence:** v1: 30% of detections were cloud-flagged, carrying 18% of RH.
  v3: the rule clears 2.3% (89/3,929) of cloud-flagged detections. Strict vs
  v1 rule changes matched totals by 2% (8.35 vs 8.19 BCM/yr).
- **Status:** active; footprint-scaled window is fix 5 of review 2.

### D17 Duplicate records (v3)
- **Choice:** same satellite and site, consecutive gaps < 10 s → one record,
  smallest `dist_m`, earliest on ties.
- **Evidence:** 4,715 groups merged; review 2 B9: 11 groups keep the
  non-detection.
- **Status:** superseded by D26 (v4).

### D18 Nightly value and status (v3)
- **Choice:** `rh_mw` = mean RH over strict-clear passes with clear
  non-detections as 0 (form of rs2021 Eq. 3 and MYC25Q4). Status precedence
  `flaring` > `flaring_cloud` > `clear_no_flare` > `cloudy` > `not_observed`;
  `flaring_cloud` = a detection under a non-isolated cloud flag and no clear
  detection that night.
- **Evidence:** review 2 B10: 218 of 1,634 `flaring_cloud` nights also have a
  clear non-detection (`rh_mw` = 0); README step 6 said `cloudy`.
- **Status:** superseded by D27 (v4).

### D19 Volume conversion (v1)
- **Choice:** BCM/yr = 0.0116 × mean RH (MW), the MYC25Q4 Zenodo slope. No
  atmospheric correction, no satellite cross-calibration.
- **Alternatives:** 0.0124 from the en2025 text (80.846 MW per BCM/yr).
- **Why:** MYC25Q4 applies 0.0116 to VNF v3.0 top-of-atmosphere RH75%, the
  quantity computed here.
- **Status:** active.

## Validation and execution

### D20 Validation against MYC25Q4 2025 (v3)
- **Choice:** one-to-one matching within 2 km, shortest pairs first; period
  value = pooled mean over all strict-clear passes; per-pass detection
  frequency vs MYC `n_det / n_clear`.
- **Evidence:** review 2 B11 (greedy pairing mis-pairs site 989/983), B12
  (definitions differ: 0.269 own-site strict vs 0.509 like-for-like vs MYC
  0.628).
- **Status:** superseded by D28 (v4).

### D21 Execution and caching (v1)
- **Choice:** `ProcessPoolExecutor` (h5py serialises threads); one CSV per
  granule under `overpasses_<PIPELINE_VERSION>`; up to 3 passes retry failed
  granules.
- **Evidence:** v3, 30 nights: 19,443 granules, 167 GB, 105 min with 12
  workers on a laptop, about 26 MB/s.
- **Status:** active.

### D22 Spectral fit guards (v4)
- **Choice:** Keep M10 or M11. Above reduced χ² 9, try every removable band and take the
  converged leave-one-out refit with the largest decrease in reduced χ²; stop at three
  bands or no improvement. Keep the 10% model error, noise floor 1e-6, χ² limit 9, 3,500
  K upper bound and ESF (0,1]. Two-band fits require 1,200–2,500 K; every flare fit
  requires ≥1,200 K. Reject failed convergence and remaining high χ². Pin rejected
  spectra to 1,810 K from valid positive M10 with ESF ≤1, otherwise leave unfitted.
  Flags record the first failed guard.
- **Alternatives:** Negative-residual elimination; two-band residual acceptance; a
  higher temperature floor.
- **Why:** The observed site-3802 spectrum fits 1,699 K and 124 MW after removing M7;
  M7+M8 alone gave 916 K and 3,127 MW. Keep the requested temperature defaults pending a
  larger validation sample.
- **Evidence:** Dev night: 0/378 retained fits below 1,200 K; 48 two-band fits span
  1,334.2–2,458.9 K; 13/378 fits pin (9 range, 3 low temperature, 1 bounds). The
  observed site-3802 regression recovers 1,699.2 K and 124.3191 MW (n=1 spectrum).
- **Status:** superseded by D33 (v5).

### D23 Peak-anchored RH75% (v4)
- **Choice:** Threshold at 0.75 × the M10 peak pixel RH. Sum fitted pixels, including
  constrained M10 pins, only through 1.5 × peak RH inclusive. Count excluded pixels.
  Record peak RH and number summed; at a multi-source site RH is summed, peak RH
  describes the strongest source, and n_rh75 sums selected pixels over retained copies.
- **Alternatives:** Highest-RH threshold; no cap; discard all pinned pixels.
- **Why:** A lower-M10 wing with much higher inferred RH is inconsistent with the source
  fit. Keep the requested 1.5× cap without tuning it from one night; the fit guards and
  peak anchoring remove the reviewed 1,776 MW failure.
- **Evidence:** Dev night: 496 pixels summed across 402 sources, 0/1103 fitted pixels
  cap-excluded. Site 3804 falls from 1,775.8899 to 71.1207 MW (n=1 case); no cap
  exclusion occurs there. The synthetic cap test excludes 1/3 fitted pixels.
- **Status:** superseded by D37 (v5).

### D24 Separate connected maxima (v4)
- **Choice:** Within each selected scan component, retain all M10 maxima passing the
  existing same-scan ±2 glow test. Sort by descending M10 then row/column; suppress
  equal-height summits within two pixels. Assign every component pixel to its nearest
  retained peak in Euclidean row/column distance; ties follow peak order.
- **Alternatives:** Watershed on M10; keep one maximum; geodesic pixel assignment.
- **Why:** Nearest-peak partition is deterministic and needs no new dependency. It
  separates the seven-pixel [10,3,2,1,2,3,11] bridge without changing the glow window.
- **Evidence:** Dev night: 9 unique confirmed components split among 378 detection
  records; 22 sites have multiple sources (24/378 records). The seven-pixel bridge test
  produces two sources (n=1 component).
- **Status:** superseded by D35 (v5).

### D25 Merge scan copies and sum distinct sources (v4)
- **Choice:** Union confirmed peaks in adjacent 16-line scans when geodesic peak
  distance is strictly less than 0.5 × min(sqrt(area_a), sqrt(area_b)). Areas use the
  existing same-scan geolocation footprint. Allow transitive chains. Keep the copy with
  greatest RH75%, first on ties, including its fit and peak location. Then assign to the
  nearest retained GFW site within 2 km. Sum distinct sources; strongest RH75% source
  supplies fit, area and peak fields. n_clusters counts confirmed pre-merge copies
  assigned through their merged source.
- **Alternatives:** Directional footprint ellipse; average footprint widths; max cluster
  per site; assign before merging.
- **Why:** The equivalent-square width makes the paper half-pixel rule explicit. Taking
  the smaller width avoids enlarging the merge radius using one distorted footprint. No
  new view-angle correction is introduced.
- **Evidence:** Dev night: 413 confirmed clusters become 402 assigned sources, merging
  11 copies; 24/378 detection records contain two sources. The under/over-half-pixel and
  three-copy-chain tests all pass (n=3 cases).
- **Status:** superseded by D38 (v5).

### D26 Detection-first duplicate selection (v4)
- **Choice:** Keep the same satellite/site consecutive-gap <10 s groups. Prefer detected
  records, then smallest dist_m, then earliest time, with stable input order for exact
  ties.
- **Alternatives:** Distance-first selection; highest RH.
- **Why:** A detection is evidence of a lit flare even when the neighbouring-granule
  site pixel is closer.
- **Evidence:** Dev night: 148 duplicate rows removed, 0 additional detections from the
  new preference (n=27446 retained observations). The synthetic two-row detection/non-
  detection test retains the detection.
- **Status:** active (v4).

### D27 Nightly status precedence (v4)
- **Choice:** Keep flaring > flaring_cloud > clear_no_flare > cloudy > not_observed. RH
  remains the strict-clear mean, including clear zeros. A flaring_cloud night can
  therefore have RH 0. Keep site_nights columns unchanged.
- **Alternatives:** Clear non-detection overrides cloudy detection; treat every
  detection as clear.
- **Why:** A detection under cloud is evidence the flare was lit. This precedence
  already existed in code; the README contradicted it.
- **Evidence:** Dev night: 121 sites have strict-clear flaring and 180 include cloud
  detections (n=7637 sites). The existing mixed cloudy-detection/clear-zero test passes;
  site_nights columns are unchanged.
- **Status:** active (v4).

### D28 MYC-centred detection frequency (v4)
- **Choice:** Keep existing validation keys and own-site strict-clear frequency. Add
  per-matched-flare detections at any GFW site within 2 km divided by strict-clear-or-
  detected passes at those sites. Group same-satellite consecutive times <10 s into one
  pass; any site may supply detection or clearing. Keep SZA ≥97° and the requested
  local-night period. New summary medians use matched persistent flares with at least
  one such pass, including those without a strict-clear own-site pass. Keep shortest-
  first one-to-one matching and the 2025 reference.
- **Alternatives:** Pool nearby site rows independently; own-site only; redefine
  existing JSON keys.
- **Why:** Counting a satellite pass once avoids counting nearby GFW structures as
  independent observations. The new denominator matches the MYC n_det/n_clear
  interpretation; matching optimization remains deferred.
- **Evidence:** Dev night: old median 0.000 (n=182 persistent site rates); MYC-centred
  median 0.667 (n=231), pooled 53.5% (357/667 passes). V3 new-definition median is 0.667
  (n=229), pooled 52.2% (343/657).
- **Status:** superseded by D40 (v5).

### D29 Peak and diagnostic outputs (v4)
- **Choice:** Keep row/col and dist_m at the site pixel. Add strongest-source
  peak_row/col/lat/lon/dist_m, n_sources, rh_peak_mw and n_rh75. Count cap exclusions
  and merged copies over all assigned pre-merge clusters. Track unique confirmed split-
  component IDs per granule, so components shared across sites can be counted once.
  Pinned pixels count as fitted constrained estimates.
- **Alternatives:** Overwrite site coordinates; count discarded copies as retained RH
  pixels.
- **Why:** The site location is needed for cloud clearing; the flare peak is needed to
  inspect association. Diagnostics distinguish observed candidates from the copies
  retained for RH.
- **Evidence:** Dev night: peak fields accompany all 378 detection records; 402 sources,
  413 pre-merge clusters, 496 retained RH pixels and 1,103 fitted pixels are
  represented. Nine split components are counted once by granule ID.
- **Status:** superseded by D39 (v5).

### D30 Empty coverage (v4)
- **Choice:** Skip a satellite when CMR produces no site-bearing slices for the
  requested dates.
- **Alternatives:** Fail on missing slice_start; fabricate jobs.
- **Why:** The requested November regression job build encountered empty SNPP coverage.
  Returning no jobs for that satellite preserves the normal NOAA-20 job selection.
- **Evidence:** Nine requested regression job builds cover ten site cases; empty SNPP
  coverage previously stopped a November build (n=1 observed failure). All ten
  comparisons now complete. The network-free all-empty-coverage test passes.
- **Status:** superseded by D41 (v5).

### D31 Development scope and checks (v4)
- **Choice:** Use v4 cache/output paths, the nine requested NOAA-20 granules and only
  the local night 2026-10-29 with 12 workers. Preserve v3 outputs. Keep the original
  comparison versions and add v4. Update existing tests only for reason-bearing pins, v4
  cache naming, peak RH threshold and distinct-source sums.
- **Alternatives:** Rerun 30 nights before review; replace historical outputs.
- **Why:** The observed spectrum fixture stores radiance/noise numbers; synthetic tests
  cover source partition, copy chains, temperature guards, deduplication and pass-level
  frequency without network.
- **Evidence:** All 1,343/1,343 dev-night granules succeed; all ten requested
  site/granule comparisons complete. All 51 tests, Ruff lint/format, configured pre-
  commit hook and whitespace checks pass. Eight same-night caches were reprocessed
  solely to finalize split diagnostics.
- **Status:** superseded by D42 (v5).

### D32 Merge scan copies closer than one pixel (v4)
- **Choice:** adjacent-scan peaks merge when closer than 1.0 × the smaller
  sqrt(footprint area) (`granule.BOWTIE_MERGE_PIXELS`), replacing D25's 0.5.
- **Alternatives:** rs2019's half-pixel spacing (D25); a row/column rule
  (same column ±1, rows 8–9 apart).
- **Why:** D25 sums distinct sources at a site, so a bow-tie copy left unmerged
  is counted twice. Copies of one flare at the swath edge sit 7–9 lines apart
  in adjacent scans, in the same column ±1, with comparable temperature and RH,
  and their peak centres are 0.50–0.83 pixel apart (12 pairs in SNPP
  `d20261029_t0703487` and NOAA-21 `d20261029_t2316490`). rs2019 does not say
  which pixel dimension "half a pixel" refers to.
- **Evidence:** dev night 2026-10-29, half-pixel → one-pixel: copies merged
  11 → 34; sites with more than one source 22 → 4; swath-edge share of
  multi-source records 53% (25/47) → 12% (1/8), against 34% of all
  detections; RH of the 47 half-pixel multi-source records was 663 MW vs
  413 MW in v3 for the same records; summed strict-clear site means 470.0 →
  453.5 MW (final v3: 450.3). Regression granule `d20261102_t2119529`: the
  4279/4280 copy pair (both about 1 km from MYC `s_9421`, persistent,
  0.727 BCM/yr, which is 172 m from site 4279) becomes one 111 MW source at
  4279 instead of 111 MW at 4279 plus 72 MW at 4280. Granule
  `d20261022_t2126021`: the 4452/4531 pair becomes one 49.6 MW source at 4531,
  nearest the stronger copy; MYC flares there are below 0.001 BCM/yr.
- **Status:** superseded by D36 (v5).

## V4 development evidence (2026-11-08)

Only local night 2026-10-29 was run in full: 1,343/1,343 granules succeeded,
about 11.4 GB, 7.7 minutes with 12 workers. The nine regression granules add
ten site comparisons (`dev_v4.py`, outputs `outputs/dev_v4/regressions.csv`).
Results below use the one-pixel merge (D32); the half-pixel outputs are kept
with a `_halfpixel` suffix in `outputs/dev_v4/`.

Baseline: `v3` is the final v3 code, taken from the 30-night run for this
night. `outputs/dev_v3/` was produced by round-3 code before the 10% model
error (D12) and is reported as `v3_dev` in `comparison.json` only; it shows
103/369 eliminated fits against 4/369 in final v3.

Scope: local night, SZA ≥ 97°, then deduplication (v3 distance-first, v4
detection-first). Fit counts describe each record's strongest source.

| Measure | v3 (final) | v4 |
| --- | --- | --- |
| Detected overpasses | 369 | 376 |
| Strict-clear detections | 257 | 255 |
| Two-band fits | 53 | 48 |
| Two-band temperature range, K | 1,334–3,202 | 1,334–2,459 |
| Fits below 1,200 K | 3 | 0 |
| Sum of detected RH, MW | 1,796.8 | 1,823.5 |
| Median detected RH, MW | 1.089 | 1.121 |
| Sum of site strict-clear means, MW | 450.3 | 453.5 |
| Flaring sites, strict / including cloud | 122 / 177 | 120 / 179 |
| RH75% / peak RH, P5 / P50 / P95 | 1.00 / 1.00 / 1.99 (n=369) | 1.00 / 1.00 / 2.04 (n=376) |
| Cap-excluded pixels | not recorded | 0/1,114 |
| Components split | not recorded | 9 |
| Copies merged | not recorded | 34 |
| Sites with > 1 source | not recorded | 4 |
| Duplicate rows removed | 148 | 148 |
| Own-site strict frequency, median | 0.000 (n=182) | 0.000 (n=182) |
| MYC-centred frequency, median | 0.667 (n=229) | 0.667 (n=231) |
| MYC-centred detections / clear-or-detected passes | 343/657 | 354/667 |

| Flag | v3 (n=369) | v4 (n=376) |
| --- | --- | --- |
| ok | 364 | 359 |
| eliminated | 4 | 4 (`M7` 2, `M12` 2) |
| pinned (v3, no reason) | 1 | – |
| pinned:two_band_range | – | 9 |
| pinned:low_t | – | 3 |
| pinned:bounds | – | 1 |

The test night contains none of the large reviewed failures, so its totals
move by under 2%; the regression granules carry those cases.

The own-site median is zero because over half the matched persistent sites
have no strict-clear detection on one night (n=182). The MYC-centred median
uses all matched persistent flares with clear-or-detected nearby passes, a
different cohort; MYC reference medians are 0.589 (n=182) and 0.636 (n=231).

All cases are NOAA-20. RH in MW, temperature in K; a dash means no detection.

| Granule | Site | v3 RH / T / bands | v4 RH / T / bands | v4 flag |
| --- | --- | --- | --- | --- |
| d20261010_t2330408 | 3802 | 3,126.9 / 916 / M7+M8 | 124.3 / 1,699 / M10+M8+M11+M12+M13 | eliminated:M7 |
| d20261010_t2330408 | 3801 | 105.1 / 1,593 / all six | – (copy merged into 3802's source) | – |
| d20261011_t2311458 | 3802 | 359.0 / 1,049 / M7+M8 | 35.4 / 1,689 / all six | ok |
| d20261014_t2216245 | 4510 | 579.5 / 853 / M7+M8 | 9.7 / 1,833 / M10+M8+M11+M12+M13 | eliminated:M7 |
| d20261016_t1810529 | 5869 | 49.2 / 1,092 / M7+M8 | 5.2 / 1,931 / M10+M8+M11+M12+M13 | eliminated:M7 |
| d20261022_t2126021 | 4452 | 553.9 / 996 / M7+M8 | – (copy merged into a 49.6 MW source at 4531) | – |
| d20261027_t0758447 | 876 | 832.9 / 882 / M7+M8 | 17.2 / 1,908 / M10+M8+M11+M12+M13 | eliminated:M7 |
| d20261102_t2119529 | 4280 | 933.3 / 1,021 / M7+M8 | – (copy merged into a 111.1 MW source at 4279) | – |
| d20261102_t2258015 | 3205 | 320.0 / 935 / M7+M8 | 12.2 / 1,829 / M10+M8+M11+M12+M13 | eliminated:M7 |
| d20261103_t2240322 | 3804 | 1,775.9 / 1,616 / M10+M7+M11+M13 | 71.1 / 1,616 / M10+M7+M11+M13 | eliminated:M12+M8 |

Tests: 51 network-free tests pass (Ruff check and format pass). New tests
cover the site-3802 spectrum, the three-band floor, the two-band range, low
temperature, component splitting, one-pixel copy merging and chains, source
sums, detection-first deduplication, pass-level frequency and empty coverage.

Open risks: one night does not calibrate the temperature bounds or the 1.5×
cap (0 exclusions on the test night). Merged sources take the stronger copy's
location, which decides the site when two sites lie within a pixel of each
other. Nearest-peak partitions and the ±2 glow window can suppress unresolved
sources. M10-only triggering, greedy MYC matching (B11), cloud-window scaling,
view-angle effects and non-upstream classification remain deferred. The
30-night v4 run has not been made.

### D33 Guard-aware fit selection (v5)
- **Choice:** Evaluate every leave-one-out trial, rank full guard validity before reduced χ², then accept the first valid state at χ² ≤9. Keep the three-band elimination floor and existing bounds; export rejected temperature on pins. Two-band residual χ² remains exactly determined; its independent guard is the 1,200–2,500 K interval.
- **Alternatives:** Rank χ² first; enumerate every subset globally.
- **Why:** A greedy step can choose a valid 1,697 K state instead of the lower-χ² 1,077 K rejection. Global subset search changes more of the estimator.
- **Evidence:** The exact N1 spectrum passes at 1,696.7 K, χ² 8.28, dropping M10 (n=1 counterexample).
- **Status:** active (v5).

### D34 Final spectral band set (v5)
- **Choice:** Require at least two of M10/M11/M12/M13 and at least one of M10/M11. Compare each fitted pixel against the same v5 algorithm with this rule disabled in optional diagnostics.
- **Alternatives:** Allow M10+M7+M8; require M12/M13 specifically.
- **Why:** Two short-wave bands alone cannot constrain the flare spectrum reliably; M10+M11 remains supported.
- **Evidence:** Three forbidden synthetic sets are rejected (n=3); accepted M10/M11 fits retain the two-band range.
- **Status:** active (v5).

### D35 Saddle prominence (v5)
- **Choice:** Retain a secondary M10 maximum only when its maximum-bottleneck 8-connected path to any retained stronger maximum has saddle/peak ≤0.5. Partition by nearest retained peak. Record the ratio; the primary maximum has ratio zero by convention.
- **Alternatives:** Split every maximum; no component splitting.
- **Why:** A max-priority path search computes the same saddle as a descending merge tree. It suppresses shallow shoulders without discarding resolved peaks.
- **Evidence:** Split/no-split/boundary synthetic cases pass (n=3); split cloud counts are in V5 development evidence.
- **Status:** active (v5).

### D36 Constrained copy pairing (v5)
- **Choice:** Sort candidate pairs by physical distance. Require adjacent 16-line scans, |Δcol|≤1, distance <1.5 times the smaller sqrt(footprint), RH75% ratio <3. Join only disjoint scan sets; export every accepted edge at both endpoints. Δrow is diagnostic, without a cutoff.
- **Alternatives:** Unconstrained unions; one-pixel spacing; require Δrow 4–10.
- **Why:** Scan uniqueness enforces pair matching and permits consecutive-scan chains. Reviewers observed Δrow 4–10 but have no validated geometry bound outside that sample; a hard row cutoff risks dropping genuine copies.
- **Evidence:** Review 3 measured 45 pairs, including 1.06/1.32-pixel copies and 1.45/1.79-pixel pairs with RH ratios 121/89. Bridge, column, RH-ratio, 1.4-pixel and three-scan tests pass. Across the 21 v5 spot cases, 4/75 accepted pairs have Δrow 1–2, supporting diagnostic-only row separation.
- **Status:** active (v5).

### D37 Peak-anchored RH75% adjacency (v5)
- **Choice:** Sum confirmed pixels in the peak partition within one row and column of the M10 peak and in its scan, with inclusive 0.75× threshold and 1.5× cap. Count cap exclusions only in this neighbourhood.
- **Alternatives:** Sum the full partition; anchor at RHmax; remove the cap.
- **Why:** en2025 §4.2.1 describes adjacent pixels and RH75%=sum(RH when RH>75% RHmax). It anchors on local RHmax, gives no cap, and does not specify a 3×3 stencil. Our explicit peak-anchored RH75% is a local estimator, not an exact reproduction.
- **Evidence:** The distant eligible-pixel regression now excludes that pixel; cap and adjacency regressions pass (n=2).
- **Status:** active (v5).

### D38 Copy-group value and location (v5)
- **Choice:** Use a spherical RH-weighted centroid of copy peak positions, assign to nearest observed GFW site within 2 km, use mean copy RH75%, and retain max-copy RH and strongest-copy spectral fit. Sum group RH at a site.
- **Alternatives:** Strongest-copy position and RH; arithmetic unweighted location; sum copies.
- **Why:** Copies are repeated observations of one source; mean avoids upward selection, centroid avoids attributing the flare solely by the brighter sample. rh_mw sums distinct sources; temp_k/area_m2/bands/fit_flag describe the strongest source’s strongest copy. n_clusters counts assigned pre-merge confirmed copies; n_sources counts copy groups.
- **Evidence:** Synthetic stronger-copy-near-B/centroid-near-A assignment and mean/max RH pass (n=1). Review 3 centroid assignment changed 8/37 spot pairs.
- **Status:** active (v5).

### D39 Copy provenance and cache completeness (v5)
- **Choice:** process_granule returns (site_records, copy_records), including header-only tables for empty granules. Cache <gid>.csv and <gid>_copies.csv together, and export copies_<tag>.csv. Pair fields are semicolon-separated in edge order for chains; group IDs are scoped by satellite/granule. Keep unassigned copies with blank assigned_site.
- **Alternatives:** Only per-site rows; discard unassigned groups; a cache hit on one file.
- **Why:** Copy-level provenance makes fits, splits, assignment and merge edges inspectable. Cloud mask and eight-neighbour counts are sampled at copy peaks; per-site cloud fields retain site-pixel semantics.
- **Evidence:** Cache completeness and integration contracts pass; every confirmed copy has the declared COPY_COLUMNS schema.
- **Status:** active (v5).

### D40 MYC-centred source-radius frequency (v5)
- **Choice:** Use copy-group centroids within 2 km of MYC for detections. Form same-satellite consecutive <10 s pass groups over nearby GFW sites in the requested local-night/SZA scope. Report strict 0/1 plus isolated-cloud and detection-clears sensitivity definitions. Preserve existing keys; add det_freq_myc_strict.
- **Alternatives:** Count all detections assigned to nearby sites; own-site only.
- **Why:** Source-radius filtering prevents nearby infrastructure from admitting another flare. Strict detections require a clear pass; sensitivity adds source detections to the clear denominator. Historical outputs without copies use recorded peak coordinates where available and legacy site detections otherwise; they cannot reconstruct group centroids.
- **Evidence:** A source at 3.3 km contributes zero; a source at 1.1 km contributes only under sensitivity until its cloudy pixel has six clear neighbours (n=1 radius/cloud regression).
- **Status:** active (v5).

### D41 Empty coverage reporting (v5)
- **Choice:** Print satellite and each UTC date whose CMR result has no site-bearing slices.
- **Alternatives:** Silent skips; abort all satellites.
- **Why:** Missing coverage should be visible without fabricating observations.
- **Evidence:** All-empty mocked coverage is skipped successfully (n=1 test); live skipped dates are retained in cases.log.
- **Status:** active (v5).

### D42 V5 development scope and output isolation (v5)
- **Choice:** Version caches v5; run only local night 2026-10-29 plus the 21 requested granules, using jobs from build_jobs. Retain v4 baseline source snapshots and outputs in dev_v5; no commit or push.
- **Alternatives:** Run 30 nights; overwrite v4; use the obsolete dev_v3 baseline.
- **Why:** Final v3 comes from the existing 30-night files; the council needs reviewable evidence before a larger rerun. Additional processing of the same night is allowed only to finalize diagnostics.
- **Evidence:** Tests and measured development artifacts are listed below; all new artifacts are under outputs/dev_v5.
- **Status:** active (v5).

### D43 Guard failures on good fits pin without elimination (v5)
- **Choice:** band elimination runs only while reduced χ² > 9. A fit with
  χ² ≤ 9 that fails a physical guard (low temperature, bounds,
  non-convergence, band set) is pinned at 1,810 K from M10 with the guard as
  the reason; D33's guard-first ranking applies within elimination steps.
- **Alternatives:** D33 as first implemented, which also eliminated bands from
  good fits until a subset passed the guards.
- **Why:** dropping bands from a fit that already matches the data, until the
  temperature clears 1,200 K, selects the answer by the guard rather than by
  the data. Elimination exists to remove an outlier band from a poor fit.
- **Evidence:** main-session check, dev night 2026-10-29: 2 fits change from
  `eliminated:M11` to `pinned:low_t` (n = 372 detections); detected RH
  1,766.68 → 1,766.11 MW; summed strict-clear site means 449.07 → 448.97 MW.
  21 granule cases: 1 record changes, site 4634 in NOAA-20
  `d20261103_t2240322`, 1,271 K `eliminated:M11` 0.49 MW → `pinned:low_t`
  0.24 MW. Test `test_good_fit_failing_guard_pins_without_elimination`.
  Pre-change outputs are in `outputs/dev_v5_preD43/`.
- **Status:** active (v5).

### D44 Atomic cache writes and job manifest (v5)
- **Choice:** each granule writes its copy table, then its site table, each
  through a temporary file and an atomic rename; a granule is cached only
  when both files exist. `pipeline.main` writes `jobs_<tag>.csv` (every
  expected granule and whether it completed) and `run_summary_<tag>.json`
  (expected, complete, missing).
- **Alternatives:** schema validation of existing files; non-empty size checks
  (a header-only copy table is valid for zero-copy granules).
- **Why:** review 4 P2/P3: a run killed mid-write left a truncated site file
  that passed `cache_complete`, and totals were built after exhausted retries
  without a record of what was missing.
- **Evidence:** test `test_atomic_cache_write_and_manifest`.
- **Status:** active (v5).

### D45 Export suppressed maxima (v5)
- **Choice:** each copy row carries its component's suppressed-maxima count,
  the strongest suppressed maximum's M10 radiance and its saddle ratio, plus
  the copy's hot-pixel count `n_hot`.
- **Why:** review 4 P9: without them the D35 saddle rule cannot be audited
  from the 30-night outputs.
- **Evidence:** test `test_suppressed_maxima_exported`; smoke run on SNPP
  `d20261029_t0703487` fills the columns (18 copies, 0 suppressed).
- **Status:** active (v5).

### D46 30-night v5 run in its own output folder (v5)
- **Choice:** the 30-night v5 rerun writes to `outputs/v5_30n/`; the v3
  30-night files stay at the top of `outputs/`, with SHA-256 checksums in
  `outputs/v3_30n_checksums.sha256`.
- **Why:** review 4 (Opus): the default `out_dir` would overwrite the v3
  baseline in the git-ignored folder, and `compare_versions.py` reads v3 from
  there.
- **Status:** active (v5).

### D47 Retry, lengthen and cache job-building requests (v5)
- **Choice:** CMR granule searches and NODD S3 listings go through
  `nodd.get_with_retries`: 4 attempts, backoff 5/10/20 s, retrying timeouts,
  connection errors, HTTP 429 and 5xx; other 4xx fail at once. The CMR read
  timeout is 300 s. CMR responses for UTC dates at least 3 days old are cached
  as JSON in `outputs/cmr_cache/<short_name>/<date>.json` (atomic write);
  newer dates are always re-queried.
- **Alternatives:** caching NODD listings as well (they answer in seconds, so
  not needed yet); deriving footprints from NODD geolocation files instead of
  CMR (downloads whole GEO granules).
- **Why:** the first resume of the 30-night v5 run ended in `build_jobs` on a
  single 60 s CMR read timeout (NOAA-21 queries), after 15,978/19,443
  granules were already cached. With retries, all 4 attempts on the same
  query timed out again. On 2026-11-08, VJ203MOD searches took 64–84 s
  (count-only 45 s; by concept ID 64 s; by provider 74 s), while VJ103MOD
  took 3 s. Job building for 31 days needs 62 VJ203MOD searches, so an
  uncached run spends about an hour there. CMR footprints of past dates do not
  change. Warming the cache with 8 parallel requests took about 10 minutes
  for all 93 product-days.
- **Evidence:** `outputs/v5_30n/run_attempt2.log`, `run_attempt3.log`; tests
  `test_get_with_retries`, `test_cmr_cache`.
- **Status:** active (v5).

### D48 Gate 7: flame-shape diagnostic, not a 0.8–1.25 band (v5)
- **Choice:** the edge/nadir RH check (review 4, gate 7) is reported as a
  diagnostic against the flame-shape model of Elvidge et al. 2016 §3.6,
  averaged over each geometry zone, with the H/R implied by the observed
  ratio. It is computed at persistent MYC sites detected on ≥ 50% and ≥ 80%
  of strict-clear passes at both geometries, with ≥ 3 clear passes per
  geometry. The original 0.8–1.25 band stays on record as failed. No
  view-angle normalisation is applied to RH.
- **Alternatives:** the 0.8–1.25 band of review 4 (contradicts en2016 §3.6,
  rs2021 §2.4 and en2025 §4.2.2); a fixed 1.4–1.5 reference (the first
  version of this entry, which compared 50–70° zenith with exactly 0°);
  side-view source adjustment (en2016 §3.6, rs2021 §2.4); an offline
  view-angle normalisation from `peak_col` (deferred; deterministic from the
  cached columns).
- **Why:** flames are taller than wide (H/R 1–4, mean 1.6), so apparent area
  grows toward the side view at stable temperature (en2016 §3.6). The
  operational calibration (en2025, John Zink tests at nadir, medium and side
  views) has no view-angle term, so MYC volumes carry the effect. The gate's
  nadir zone (columns 1,008–2,191) spans 0–36° zenith and the edge zone 53–70°.
  Averaged over the zones, the model gives 1.25, 1.36, 1.47, 1.56 and 1.78
  for H/R 1.4, 1.6, 1.8, 2.0 and 2.5 (review 5, O9). With H/R free, the model
  fits ratios from 1.0 to about 2.5, so it cannot serve as a pass band.
- **Evidence (30 nights, requested nights only, site-pixel column):** v5 mean
  of copies 1.53 (n = 42 sites at ≥ 80%; implied H/R 1.94), max 1.60, sum
  1.72; v3 1.62 (n = 43). The excess persists in single-source records without
  merged copies, 1.49 (n = 42, Opus). Edge temperature is 3% above nadir
  (1,789 vs 1,729 K, n = 42), which alone gives ×1.12 through T⁴; en2016
  reports stable temperature, so part of the excess has another cause
  (candidate: the uncorrected atmospheric path, 2× at 61°). Detection per
  clear pass is 0.29 at the edge and 0.39 at nadir (v3), which biases the
  all-detection ratio upward. RH75% pixel counts barely differ by geometry
  (1.24 edge vs 1.09 nadir copies, v5 dev night).
- **Status:** active (v5); view-angle normalisation deferred. Review 5 accepted
  the flame-shape basis (both reviewers) and the corrected zone-mean numbers
  (Opus).

### D49 Process the oldest granules first (v5)
- **Choice:** `build_jobs` sorts jobs by granule start time across satellites,
  so a run processes the oldest nights first.
- **Alternatives:** per-satellite order (SNPP, NOAA-20, NOAA-21), as before;
  running the first nights as a separate invocation.
- **Why:** NODD keeps about 91 days; on 2026-11-08 the oldest backfill night
  (2026-08-09) is at the deletion edge. With per-satellite order, NOAA-21's
  oldest granules would start hours after launch (review 5, O13).
- **Evidence:** test `test_build_jobs_oldest_first`. Results do not depend on
  job order.
- **Status:** active (v5).

### D50 NODD backfill scope and runner (v5)
- **Choice:** after review 5's `BACKFILL: GO`, the v5 code runs local nights
  2026-08-09 to 10-05 into `outputs/v5_backfill/` through `backfill.py`. The
  runner warms the CMR cache in parallel, retries `pipeline.main` up to 5
  times, validates, and runs detached with `status.json` showing the phase.
  Started 2026-11-08 22:37 UTC with code `b0bd15d` plus `backfill.py`.
  Together with `outputs/v5_30n/` this covers 88 nights (08-09 to 11-04).
  Nights after 11-04 run as their own segment once at least 3 days old
  (review 5, required item 2).
- **Alternatives:** adding per-copy band radiances to the copy table first
  (review 5, O11; about one night lost to NODD expiry while implementing);
  starting at night 08-10, the first full NOAA-20/21 night.
- **Why:** the user chose to start at once and iterate on design meanwhile.
  NODD expires one night per day from the oldest end.
- **Evidence:** NODD listing at 22:35 UTC: UTC 2026-08-09 holds 21 (NOAA-20)
  and 22 (NOAA-21) M10 files and none for SNPP; UTC 08-10 is complete for
  NOAA-20/21 and holds 170 SNPP files; SNPP is complete from UTC 08-11. Night
  08-09 is therefore partial on all satellites, and night 08-10 is partial
  on SNPP. Any change to fits or to the RH75% stencil after these UTC dates
  expire needs the NASA archive (O11).
- **Result:** finished 2026-11-09 02:21 UTC after 224.5 min; 36,875/36,875
  granules, 328 GB downloaded. 80 granules failed on pass 1 with NOAA
  connection resets and all 80 completed on pass 2. 442,946 site-nights
  (7,637 sites × 58 nights): clear, no flare 64.4%; cloudy 32.2%; flaring
  1.75%; flaring under cloud 0.64%; not observed 0.96% (4,255, of which
  4,223 on partial night 08-09). Flaring sites per night, nights 08-10 to
  10-05: median 130, range 85–195 (n=57); the 30-night run has median 130,
  range 104–182 (n=30). Summed radiant heat at flaring sites per night:
  median 688 MW, range 324–963 (n=57), against 548 MW, 238–709 (n=30) for
  10-06 to 11-04. MYC25 sites matched within 2 km and detected at least
  once: persistent 86.0% (306/356; 30-night run 79.2%, 282/356),
  intermittent 51.2% (87/170; 40.9%, 70/171), sporadic 19.4% (13/67;
  16.4%, 11/67), undetected 4.2% (5/118; 2.5%, 3/118). Log-RH correlation
  with MYC25 r = 0.56 (n=406; 30-night 0.51, n=363), median ratio 0.67
  (0.63). Per-pass detection frequency at persistent sites, like-for-like
  clears: 0.50 (22,046/44,654) against 0.51 in the 30-night run and 0.63
  in MYC25.
- **Status:** active (v5).

### D51 Viewer published on GitHub Pages, with the explainer as a tab (v5)
- **Choice:** the flaring viewer moves from a private Claude artifact to
  GitHub Pages at https://skytruth.github.io/offshore-methane/, built from the
  30-night v5 run. `build_viewer.publish_pages` pushes the built file as a
  single parentless commit (`index.html`, `.nojekyll`) to `gh-pages`, so the
  branch never accumulates copies of the inlined data. The page carries a
  "How it works" tab rendered from `docs/explainer.md`, with the SVG figures
  from `docs/explainer/make_figures.py` inlined; `#how-it-works` opens it.
- **Alternatives:** keep the Claude artifact (private; user asked to move);
  commit the built page to `main` or build it in GitHub Actions (the data are
  git-ignored run outputs, so CI cannot build it).
- **Why:** the user asked for a deployed GitHub page. The repository is public,
  so the page is public. It shows EOG MYC25Q4 values (CC BY-NC 4.0) and
  site positions derived from GFW fixed infrastructure, both attributed in the
  page footer.
- **Status:** active; D52 changes the build and the page. The Claude artifact
  (v3 data) was deleted on 2026-11-09 at the user's request.

### D52 Website built for a growing night series (v5)
- **Choice:**
  - *Data.* `build_viewer.main` joins every run segment under `outputs/v5_*`
    into one night series. Where segments overlap, the later segment wins;
    a night no segment covers shows as not observed. The build writes a
    folder, `outputs/site/`: `index.html` with the per-night data inlined,
    `passes/<site_id>.json` for each site that flared (fetched when the site
    is opened), and `img/` for the explainer chips. `publish_pages` pushes the
    folder as one parentless commit. Each site carries a status string (one
    code per night), clear-sky radiant heat only for flaring and
    lit-through-cloud nights, and, for sites that flared, clear passes per
    night as one base-36 digit.
  - *Flare list.* Three orders: Strongest (mean clear-sky radiant heat, or the
    night's value in single-night mode); Most consistent (share of clear
    nights with a flare, ranked by the lower end of its 95% Wilson interval);
    Most clear views (clear satellite passes in the selected nights). On a
    single night the list holds that night's flaring sites, ordered by their
    figures over the selected range. Pages of 25 rows (10 on screens up to
    900 px) with "Show more".
  - *Timeline.* A band under the bars shows the share of all sites seen in
    clear sky each night. Date labels follow the space per night: every
    night, Mondays, months, or quarters. Buttons set the last 7, 30, 90 or
    365 nights (those shorter than the record) or All; ←/→ step a night,
    Home/End jump to the range ends. A night is flagged partial when fewer
    than 90% of the median number of sites were passed over.
  - *Small charts.* Line sparklines become one column per night (height
    relative to the site's maximum) over a strip coloured by status;
    adjacent nights share a column when nights outnumber pixels.
  - *Site detail.* An overview of every night carries a draggable window of
    14 nights (the whole range when it is 21 nights or fewer); the pass chart
    below covers the window. Passes are deduplicated and classed clear by the
    same rule as the nightly values (`dedupe_overpasses`, `strict_clear`), so
    a detection under cloud that does not count is drawn as a ring. The time
    axis is the site's local solar time, so each band is that site's night.
  - *Controls* sit next to what they act on (user request, 2026-11-09):
    Play at the left of the timeline bars; Range mean / Single night above
    the four statistics, with the period they cover; All sites / Flaring
    only in the map legend (which then drops the clear and cloudy keys);
    the percentile filter at the top of the flare list, noting that it also
    hides flares on the map and in the totals. The separate night slider is
    removed: the bars, their playhead and the arrow keys select nights.
  - The tooltip and detail panel said EOG flares were matched "within 1 km";
    the matching radius is 2 km (D20), and the text now says so.
- **Alternatives:** inline the pass rows (page about 8.6 MB at 88 nights,
  growing about 67 KB per night); keep line sparklines (0.8 px per night at
  72 px and 88 nights); rank consistency by the raw share (a site flaring on
  its only clear night would top the list at 100%, 1/1); count clear nights
  for "clear views" (ties at the night count).
- **Why:** the user asked for the three list orders and a site designed for
  90 nights and more.
- **Evidence:** 88-night build (segments `v5_backfill` and `v5_30n`):
  `index.html` 2.7 MB (0.58 MB gzip), 736 pass files 5.9 MB. A synthetic
  365-night page (the 88 nights repeated) is 5.3 MB; in headless Chrome a
  preset click (range statistics plus redraw) took 12 ms against 8 ms at 88
  nights, and labels switched to months. Combined validation (MYC25, 2 km,
  one-to-one): detected on at least one clear night at persistent sites
  88.5% (315/356), intermittent 57.3% (98/171), sporadic 23.9% (16/67),
  undetected 5.9% (7/118); log-RH r = 0.59.
- **Status:** active (v5).

### D53 Website restyled to the SkyTruth brand, in dark mode (v5)
- **Choice:** the page follows the SkyTruth Brand Style Guide (2026) and keeps
  a dark theme (user request, 2026-11-09). Surfaces use the Blue family
  (Blue8 page, the guide's dark background; Blue7 panels; Blue5 rules);
  flares use the Orange family (Orange8 to Orange1, the guide's flaring
  palette); Orange6 is the highlight for buttons, tags and the short rule
  under each heading; links are off-white with an Orange6 underline, as the
  guide sets for dark backgrounds. Selection uses Forest2. Headings use
  Tiempos Fine and body text PP Neue Montreal where installed, falling back
  to the guide's general-use pair, EB Garamond and Roboto (Google Fonts).
  The header carries the SkyTruth logo in the dark-blue colourway (white
  wordmark, Blue5 ring) prescribed for Blue8 backgrounds, at the 8rem web
  minimum, linked to skytruth.org; the vector comes from skytruth.org's
  header (`assets/skytruth-logo-dark.svg`). Two of the guide's graphic
  elements appear: the data-points dot pattern (header and explainer, both
  hidden on narrow screens) and the grid overlay, drawn as a faint map
  graticule whose spacing follows the zoom. The explainer figures and image
  chips use the same palette. Controls, data and interactions are unchanged.
- **Alternatives:** the guide's light tool-page style (white background);
  the user asked to keep dark mode. Loading the licensed web fonts from
  skytruth.org (their licence scope for this domain is unknown).
- **Why:** user request to follow the brand guidelines.
- **Status:** active (v5).

### D54 Radiant-heat filter ranks once per range; opened site always drawn (v5)
- **Choice:** the percentile filter ranks flares by mean clear-sky radiant
  heat over the selected nights and keeps that set on single nights and
  during playback; before, single nights and playback ranked by each night's
  value, so the set changed every night. The site open in the detail panel
  is drawn whatever the filters, with its state for the night; its ring
  follows the drawn dot and is dashed on nights with no pass. The control is
  labelled "Radiant-heat filter" and states that it uses mean radiant heat
  only, not the list order. Clicking a dot on the map centres it at the
  current zoom; choosing from the list still zooms in. The pipeline version
  moves from the header to the citation line.
- **Alternatives:** keep a per-night cut-off and exempt only the opened site
  (the other shown flares would still change from night to night).
- **Why:** user report, 2026-11-09: with the filter on, an opened site
  vanished during playback on nights below that night's cut-off, leaving an
  empty selection ring that changed size with the night's value.
- **Status:** active (v5).

### D55 Larger flare circles drawn beneath smaller ones (v5)
- **Choice:** map flare circles are drawn largest first, so smaller flares sit
  on top, each with a dark outline. Hover and click pick the nearest site
  centre, so the order does not change what they select. The pass chart in
  the site panel gets a y-axis title, "Radiant heat (MW)".
- **Alternatives:** largest on top (the previous order: small flares beside a
  large one were hidden, understating how many sources share an area);
  semi-transparent fills (colour encodes radiant heat, and blending would
  change it).
- **Why:** user question, 2026-11-09; drawing large proportional symbols
  first is the usual practice in thematic cartography (Slocum et al.,
  Thematic Cartography and Geovisualization).
- **Status:** active (v5).

### D56 Explanations move into (i) tooltips (v5)
- **Choice:** method text that sat inline moves into (i) buttons that show
  on hover, keyboard focus or tap (Esc or a tap elsewhere hides them): the
  timeline (what bars and band show, click/drag/arrow keys), the flare-list
  order (metric definitions; the caption keeps the metric and period), the
  radiant-heat filter, the flaring-sites, radiant-heat and gas-flared tiles,
  the site panel's pass chart (dragging the window, local solar time) and
  the EOG comparison (matching method; results become three short rows).
  Tip text is built when shown, so it names the current range and order.
- **Alternatives:** native `title` attributes (no touch or keyboard access,
  delayed, unstyled).
- **Why:** user request, 2026-11-09.
- **Status:** active (v5).

## V5 development evidence (2026-11-08)

This section is the current development evidence; the V4 section above is retained
as historical evidence. Only local night 2026-10-29 was run in full. All 1,343/1,343
granules succeeded with both cache files. The first pass fetched 11.44 GB in
7.8 minutes (12 workers; see night.log for exact totals). Finalizing the
band-rule diagnostics reprocessed 172/1,343 same-night granules, 4.93 GB in
2.2 minutes. This did not add another night. The 21/21 requested cases each ran
v4 and v5 using process_granule and build_jobs. No 30-night v5 run was made.
All artifacts are in outputs/dev_v5. Baseline source snapshots come from HEAD
(d6c50b5); the case baseline is run directly rather than inferred from v3.

The table uses local evening date, SZA ≥97°, and detection-first deduplication
for v4/v5. Final v3 comes from the 30-night file with its original deduplication.

| Measure | v3 (final) | v4 | v5 |
| --- | --- | --- | --- |
| Detected overpasses | 369.00 | 376.00 | 372.00 |
| Detected RH, MW | 1796.77 | 1823.48 | 1766.11 |
| Sum strict-clear site means, MW | 450.33 | 453.50 | 448.97 |
| Strict-clear detections | 257.00 | 255.00 | 256.00 |

V5 strongest-source fit flags (after D43), n=372: ok 355;
eliminated:M12 2; eliminated:M7 2; pinned:two_band_range 9; pinned:low_t 3;
pinned:bounds 1. Pinned share is 3.5% (13/372). Accepted forbidden band sets
are 0/372 in the local night and 0/824 in the complete v4 development granules.
A controlled v5 refit with only D34 disabled changes 4 confirmed pixel fits in
the 172 audited granules, including 0/1,046 confirmed copy-peak fits. These
pixel changes do not alter the final local-night totals. n_band_rule_changed
is a granule-wide count repeated on site rows; aggregate its maximum per
satellite/granule, not its sum over sites.

There are five unique split components represented by ten local-night detection
records. Their non-isolated-cloud share is 80% (8/10); v4 had nine components.
Four local-night records at four sites have multiple sources; their swath-edge
share is 0% (0/4), against 29.8% (111/372) of all detections. Edge means peak
column <640 or ≥2560. There are 32 merged copies represented in local-night
records. Across all processed granules, 1,046 copies form 954 groups, with
92 merged groups and 92 accepted pair edges. Accepted Δrow spans 1–9; 13/92
accepted pairs lie outside 4–10 lines. This further supports D36's diagnostic-
only Δrow choice. Component IDs and group IDs are scoped by satellite/granule.

MYC-centred strict frequency is 46.1% (263/571 passes), median 0.400 across
188 persistent flares with strict-clear passes. Detection-clears sensitivity
is 53.5% (354/662 passes), median 0.667 across 229 persistent flares. The latter
retains det_freq_like_for_like and has the explicit alias
det_freq_myc_detection_clears; strict is det_freq_myc_strict. Own-site strict
median remains 0.000 (n=182). Recomputed historical v4 peak-radius sensitivity
is 52.8% (350/663 passes), median 0.667 (n=230), rather than the old unfiltered
354/667 metric. V3 lacks source positions, so its 343/657 pass metric remains
site-based; historical frequencies cannot reproduce v5 group-centroid filtering.

### Requested granule cases

RH totals below sum all site records in each selected granule, including cloud
and outside-local-night records. They differ in scope from the local-night table.
case_records.csv contains every detected v4/v5 site record, case_groups_v4.csv
and case_groups.csv contain the source-group tables, and cases/ contains raw
v4 constituent/group snapshots and v5 copies, including peak fit/cloud data.

| satellite | granule | detections_v4 | detections_v5 | rh_v4 | rh_v5 | named_sites |
| --- | --- | --- | --- | --- | --- | --- |
| NOAA-20 | d20261010_t2330408 | 11 | 10 | 191.11 | 169.98 | 3752: 45.00 → 37.54; 3802: 124.32 → 114.71 |
| NOAA-20 | d20261011_t2311458 | 29 | 28 | 551.27 | 503.58 | 3752: 22.20 → 22.20; 3802: 35.42 → 34.77; 4197: 21.74 → 21.74; 4267: 77.14 → 58.38; 4279: 103.05 → 94.30 |
| NOAA-20 | d20261013_t0043475 | 16 | 16 | 27.74 | 26.9 | nan |
| NOAA-20 | d20261014_t0704356 | 12 | 11 | 134.61 | 92.27 | 971: 21.65 → 21.65 |
| NOAA-20 | d20261014_t2216245 | 10 | 10 | 61.62 | 51.39 | 4267: 21.14 → 10.92 |
| NOAA-20 | d20261016_t1810529 | 9 | 9 | 8.76 | 8.69 | nan |
| NOAA-20 | d20261021_t2144572 | 18 | 18 | 171.97 | 164.88 | 4197: 2.42 → 2.42; 4267: 17.66 → 13.06; 4279: 50.95 → 50.95 |
| NOAA-20 | d20261022_t2126021 | 17 | 17 | 241.23 | 201.86 | 4197: 1.39 → 1.39; 4267: 26.43 → 26.43; 4279: 129.40 → 97.15 |
| NOAA-20 | d20261022_t2305365 | 38 | 37 | 370.92 | 357.21 | 3752: 28.71 → 22.12; 3802: 56.30 → 56.30; 4197: 5.96 → 4.90; 4267: 18.14 → 18.14; 4279: 88.65 → 88.65 |
| NOAA-20 | d20261027_t0758447 | 2 | 2 | 20.43 | 20.43 | nan |
| NOAA-20 | d20261030_t2216381 | 10 | 10 | 82.89 | 82.89 | 4267: 37.00 → 37.00 |
| NOAA-20 | d20261102_t2119529 | 16 | 16 | 278.58 | 251.5 | 4197: 6.41 → 6.41; 4267: 34.94 → 34.94; 4279: 111.13 → 91.80 |
| NOAA-20 | d20261102_t2258015 | 1 | 1 | 12.21 | 12.21 | nan |
| NOAA-20 | d20261103_t2240322 | 39 | 39 | 392.21 | 387.25 | 3752: 6.97 → 6.27; 3804: 71.12 → 71.12; 4197: 11.49 → 11.49; 4267: 27.20 → 22.60; 4279: 68.36 → 68.36 |
| NOAA-20 | d20261105_t2204060 | 15 | 15 | 133.6 | 133.6 | 4202: 3.96 → 3.96; 4267: 19.54 → 19.54; 4279: 21.92 → 21.92 |
| NOAA-21 | d20261013_t2317594 | 7 | 7 | 196.55 | 188.84 | 3752: 24.81 → 24.81; 3801: 165.65 → 157.95 |
| NOAA-21 | d20261022_t0837523 | 8 | 7 | 77.75 | 59.32 | 957: 2.23 → 0.00; 971: 4.06 → 3.15 |
| NOAA-21 | d20261030_t0748551 | 10 | 9 | 15.67 | 13.31 | 971: 0.56 → 0.56; 981: 1.64 → 0.00 |
| NOAA-21 | d20261030_t2118195 | 14 | 13 | 330.48 | 297.51 | 4197: 11.85 → 13.79; 4202: 15.74 → 0.00; 4267: 122.73 → 122.73; 4279: 0.00 → 112.68; 4281: 127.87 → 0.00 |
| SNPP | d20261020_t2323324 | 22 | 22 | 241.03 | 227.51 | 3752: 39.14 → 39.14; 3802: 5.81 → 5.81; 4197: 0.00 → 6.64; 4267: 32.19 → 31.12; 4279: 112.35 → 101.64 |
| SNPP | d20261029_t0703487 | 10 | 9 | 203.28 | 168.9 | 971: 53.28 → 42.11 |

The named NOAA-21 d20261030_t2118195 group moves from 4281 (127.87 MW) to
4279 (112.68 MW); 4197/4202 become 13.79/0 MW from 11.85/15.74 MW. NOAA-21
d20261022_t0837523 sites 971/957 become 3.15/0 MW from 4.06/2.23 MW.
NOAA-21 d20261013_t2317594 retains the 3801/3804 copy group, reducing its
max-selected 165.65 MW to a 157.95 MW mean. SNPP d20261020_t2323324's
4279/4284 copies and separate 3752 source are preserved in case_groups.csv.
At the cloud split NOAA-21 d20261030_t0748551, detections fall 10→9 and RH
15.67→13.31 MW; site 981's 1.64 MW detection disappears when its shallow
secondary peak is suppressed. The surviving site 984 source falls 0.97→0.68 MW.
The adjacency-only effect is independently isolated in unit tests. NOAA-20
d20261030_t2216381 site 4267 retains two sources and 37.00 MW: its 1.53-pixel
separation remains outside the copy threshold.

### Pair-distance histograms

Candidate population: adjacent scans, |Δcol|≤1, separation <3 smaller-footprint
pixels, before distance/RH eligibility. Columns split the RH75% ratio. This
counts candidate edges, not copies. All processed development granules (n=104):

| bin | <3 | >=3 |
| --- | --- | --- |
| [0.0, 0.5) | 36 | 0 |
| [0.5, 1.0) | 52 | 6 |
| [1.0, 1.32) | 4 | 3 |
| [1.32, 1.45) | 0 | 0 |
| [1.45, 1.5) | 0 | 0 |
| [1.5, 2.0) | 0 | 0 |
| [2.0, 3.0) | 1 | 2 |

The 21 spot granules (n=83):

| bin | <3 | >=3 |
| --- | --- | --- |
| [0.0, 0.5) | 20 | 0 |
| [0.5, 1.0) | 52 | 2 |
| [1.0, 1.32) | 3 | 0 |
| [1.32, 1.45) | 0 | 1 |
| [1.45, 1.5) | 0 | 0 |
| [1.5, 2.0) | 0 | 2 |
| [2.0, 3.0) | 1 | 2 |

The 1.31842-pixel pair has RH ratio 1.82 and merges; the 1.44803-pixel pair
has RH ratio 120.88 and remains separate. No candidate occurs between those
two distances in the spot cases; this small sample does not calibrate a universal
1.5-pixel boundary. CSVs list each pair and its match outcome.

### Tail inspection, checks and remaining risks

Cap exclusions are 0 across 1,046 copies. Four copies have RH75%/peak >3;
rh75_tail.csv lists all four. They sum three or four eligible adjacent pixels,
not remote component pixels: SNPP d20261030_t2017091/site 4823 ratio 3.90
(four pixels, cloud 0); NOAA-20 d20261029_t2234075/site 4263 ratio 3.32
(three, cloud 1); NOAA-20 d20261030_t0704492/site 976 ratio 3.83 (four,
cloud 3); NOAA-20 d20261030_t0844236/site 992 ratio 4.00 (four, cloud 3).
These are compatible with the stencil but the two cloud-3 cases remain
candidates for diffuse cloud contamination; no cap was triggered.

All 67 network-free tests pass; Ruff check, Ruff format, configured pre-commit
Ruff hook and git diff --check pass. Updated existing tests: adjacency changes
RH expectations in test_cluster_nearest_site_rh75_and_bowtie_max and
test_peak_rh_threshold_and_cap; test_elimination_stops_at_three_and_keeps_anchor
now expects band_set for the forbidden three-band input; the bowtie chain test
uses a 1.501-pixel rejection and mean RH; process integration accepts the new
tuple/diagnostic argument and checks copy columns; main output-path test uses
v5 and a structured synthetic job. New test_v5 covers all requested counterexamples,
saddle boundary, same-scan bridge, pair guards, centroid assignment, cache
completeness, copy schema, adjacency/scan exclusion, and MYC source radius.

Defaults changed only as documented by D33–D42: guarded elimination and final
bands; saddle prominence; local RH stencil; 1.5-pixel paired merging with column,
RH and scan constraints; centroid assignment; mean RH; v5 cache version; copy
outputs and source-radius frequency. process_granule now returns a tuple;
dev_v4 is adapted to select the site table. The existing dates, cloud thresholds,
M10 trigger, fit bounds/model error, calibration and association radius remain.
Diagnostic refits roughly double fitting work when diagnostics are requested;
public NOAA reads dominate the observed run cost.

Open risks: 80% (8/10) split-tagged local-night detections remain under
non-isolated cloud; one night cannot calibrate split, cap or copy thresholds.
Subpixel distinct sources with comparable RH in adjacent scans can merge.
Site 4267 remains an unresolved 1.53-pixel two-source case. Observed-site nearest
assignment can still select the wrong structure in dense fields. Optimal MYC
matching (B11), association basins, cloud-window scaling, view-angle normalization,
M11/M12-M13 triggers and non-upstream classification remain deferred. The 30-night
rerun must inspect these distributions and the source-attribution flips before
backfill. No implementation item is intentionally deferred from this fix set.
