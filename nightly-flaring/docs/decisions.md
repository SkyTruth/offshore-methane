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
- **Status:** active. Nights older than the AWS window: NOAA's Azure copy (D63).

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
  on top. Hover and click select the circle under the pointer (D57). The
  pass chart in
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

### D57 Circle edges and pointer hits (v5)
- **Choice:** each flare circle's edge is a thin line (0.75 px under 6 px
  radius, else 1 px) in a darker shade of its own fill at 85% opacity,
  replacing a 1.5 px near-black outline. Hover and click select the topmost
  drawn circle that contains the pointer (1.5 px tolerance); only when none
  does, the nearest centre within 6 px, for dots too small to hit.
- **Alternatives:** keep the outline but lower its opacity (dark rings still
  stack in dense fields); no outline (overlapping circles of similar colour
  merge); the nearest centre within 14 px (the previous rule: a click inside
  a large circle could select a neighbour whose centre was closer).
- **Why:** user reports, 2026-11-09: outlines dominated dense fields after
  D55 put small circles on top, and clicks selected a site other than the
  circle clicked. In the Abu Dhabi field, points inside site 4452's 13.5 px
  circle now report 4452 (6/9 probes) or a smaller flare drawn over it
  (3/9: sites 4531, 4469, 4546).
- **Status:** active (v5).

### D58 Source-centred flare-out evidence cards and tiers (v5)
- **Choice:** source histories use raw cloud-0/1 night passes (SZA ≥97°), with
  one clear pass qualifying a night. Any copy-group detection within 1.5 km
  resets darkness, including unassigned/cloudy detections. Heat is the median
  clear copy-group heat within 1 km. Assignment centres within 750 m merge
  subject to a simultaneous resolved-source guard. Overlapping candidate
  spells within 1.5 km yield one card, preferring the complex with more
  initial/discovery-baseline detections; suppressed IDs are retained on it.
- **Prospective baseline:** for every scanned source-night and each spell,
  freeze the source unit's own preceding 29 nights, ending the night before.
  Eligibility is ≥12 clear nights, ≥80% with source detections, and median
  measured heat ≥3 MW. The partial first record night is excluded; evaluation
  begins after 29 full nights. Later bright sources can enter and requalify.
  Assigned-site detections never substitute for the scored source's baseline.
- **Dated heat:** log-normal/censored Viterbi states date normal/reduced heat;
  scoring uses states fitted only through the pre-spell night. Pre-spell heat
  uses ≥3 clear detections in the intersection of the prior week and the last
  dated heat state. Thus a 50→8 MW change is scored from 8 MW. A later dark
  spell is not suppressed solely for following reduced heat. Gaps with <3 clear
  nights merge into the surrounding state; ≥3 clear misses form a darkness
  row. Every detection ends that row, including a weak recovery. Table counts
  say "nights with heat measured"; darkness never includes a recovery's heat.
- **Arithmetic:** use transitions on the same source's frozen baseline clear
  opportunities, with cloudy positives resetting the previous state without
  becoming extra trials. First-miss rate a=(HM+1)/(HH+HM+2). Refit pooled
  MM/MH across eligible nonoverlapping 29-night source blocks on each build;
  r=max(0.59, Beta(MM+1,MH+1) 97.5th percentile). Tail=a×r^(k−1).
  Each scored spell has its own defined 30-night fleet window containing its
  tested start. For an ended spell of ≤30 nights, end the window at its last
  dark night; that window does not move on later builds. For an ongoing spell
  of ≤30 nights, end it at as-of. For either spell longer than 30 nights, use
  its first 30 nights and keep that window as the spell continues or ends.
  The scan family F is the sum of prospectively eligible source units on each
  available evaluation night in that window, floored at the number of those
  nights (one tested unit per available night, at most 30). This conservative
  floor includes at least the scored unit's own eligible nights, even when
  no source meets the bright rule on some nights; an unrelated empty later
  scan is never substituted. During startup, only evaluation nights after
  the initial baseline count. Archived months do not grow the penalty merely
  through record length; new thermal evidence can change eligibility and
  recalibration. Score=F×tail; budget=0.1 per spell's fleet scan rather than
  a cumulative lifetime budget.
  Display score/0.1 as a multiple of the High threshold for High/Mid, not Watch,
  which cannot satisfy the bright rule.
  On 9 November, as-of night 4 November: MM/MH=13/37; r=0.59.
  The ended 4582/4583 spell uses the frozen 5 October–3 November window,
  F=1,348 unit-nights. Its first-miss rate is 2/28; score=0.10107
  (1.01074× budget, about 1.011×) at 14 clear dark nights. The score nearly ties
  the gate: at unchanged calibration and family, 15 dark nights
  would give 0.59633× (about 0.60×). Across this record's 29 full 30-night
  windows, F ranges from 1,084 to 1,353, giving 0.81–1.01× for the same
  14-night evidence. The measured shared heat loss at 4500/4515 is the rule
  that keeps this card at Mid even under window choices that pass the score
  gate; its frozen score also narrowly fails today. No extra safety factor
  is tuned to preserve that tier. For first-miss rate 0.07 the score gate
  also needs 15 nights. Ongoing 927 uses 6 October–4 November, F=1,353:
  53.73848× at six clear dark nights and first-miss rate 1/18; it needs 14
  at unchanged calibration/family and has no comparison flare to establish
  isolation.
  These remain model sensitivities, not validated false-alarm probabilities.
- **Tiers:** High requires the recent bright rule, dated pre-spell heat ≥3 MW
  from ≥3 clear detections, ≥6 clear dark nights from ≥2 satellites, a sampled
  3 km absence check, resolved isolation and score≤0.1. Mid requires the same
  source/heat/satellite/absence checks and ≥6 clear dark nights, with shared
  loss or score above budget; missing/unresolved comparison coverage caps a
  passing score at Mid. Mid uses no independent-night score and does not drift
  with record length. Watch requires an ongoing ≥28-clear-night spell, ≥12
  earlier detected nights, ≥3 pre-spell clear detections and ≥2 satellites,
  while failing the recent bright rule. No current card qualifies as High.
- **Comparison heat:** complexes >3 km away, target excluded, at 10/50/100 km.
  A comparison flare needs ≥3 nights with measured clear heat within 1 km in
  the preceding 14 nights, median ≥3 MW. Only measured heat or qualified
  nondetections enter the spell denominator. ≥3 low nights and ≥half of those
  measured nights below one fifth of baseline establish shared loss. Overlapping
  comparison apertures count once, keeping the better-supported complex.
  Isolation requires ≥1 comparison flare within 10 km, every such flare measured
  on ≥3 nights, no shared loss, and enough non-losing coverage that even all
  unknown nights losing heat could not meet the ≥3/≥half-spell loss rule. Otherwise
  say "no comparison flare within 10 km" or "comparison heat unresolved".
- **Heat bound and search limits:** use same-night re-detection by heat class;
  tail=(1−p)^k. Exclude a class only when it and every brighter class have
  tail≤0.05. Independence and detectable-once conditioning make this a
  conditional bound, never extinction. The 3 km veto sees only clusters
  within ±2 pixels of GFW sites, not a complete 3 km search. Export the count
  of otherwise eligible vetoed spells in `source_veto_count`.
- **Presentation:** dark SkyTruth overview, tier counts, clickable SVG locator,
  ranked compact list and one detail pane; charts use a labelled spell band,
  MW linear axes, misses at zero, named comparisons with distances. Scores,
  calibration and heat history are collapsed. Passes are collapsed separately.
  Phone stacks list then detail. Payload excludes unused global heat-state
  histories and comparison series beyond 10 km.
- **Provenance:** `flareout_research/{sol,opus}_r3.md` and checked-in `.py`
  scripts in `flareout_research/scripts_{sol,opus}/`; no research caches copied.
- **Status:** active; evidence cards for review, never a venting accusation.

### D59 Public EMIT overpasses and optional review imagery (v5)
- **Choice:** public CMR EMITL2BCH4ENH and EMITL1BRAD, acquisition deduplication,
  public PNG browse links, ±14-night windows capped at as-of. Convert local
  noon-to-noon night bounds to UTC using longitude/15 hours. Before/after are
  bracketed by last-lit/recovery VIIRS observations; during lies between first
  and last clear dark passes. Between these observations label "boundary".
- **Caching:** a listing becomes final only when fetched ≥30 days after the
  window ends. Earlier snapshots refresh online; offline snapshots preserve
  scene/chip review with an incomplete listing status. Successful chip caches
  are consulted before initializing Earth Engine, and copied into the build
  even if a fresh EE connection fails. Failed imagery is retried each build.
- **Recipe:** HITL `(mean radiance_231..244 − mean radiance_256..268) /
  mean radiance_231..244`; median and robust sigma within 15 km; display
  strength=clip((index_sigma−1)/5,0,1)^0.65. True colour matches
  `notebooks/hitl_app.py::_emit_truecolor` exactly: prefer L2A reflectance
  bands 35/23/11; fall back to L1B radiance bands 35/23/11. Compute per-band
  2/50/98 percentiles within 20 km at scale 120 m (`bestEffort=True`), then
  use one shared minimum (smallest p2) and maximum (largest p98), gamma 1.8.
  Reject missing percentiles; reject L2A if any p98>1.5; reject either source
  if max(medians)/max(min(medians),1e−9)>5 or shared range<1e−6. If neither is
  usable, imagery is unavailable and retried. These are the source's actual
  guards, resolving the reviewers' disagreement about per-band stretching.
  Both chips mark the source and show a 5 km scale. Recipe `hitl-v3`
  invalidates earlier chip renders while retaining their cache artifacts.
  Match CMR v2/v3 acquisition times to EE v1 assets rather than invent suffixes.
- **Wording:** "not in Earth Engine yet (its EMIT collection ends <date>)"
  uses the measured `collection_latest_utc` for recent unavailable assets,
  without assuming future ingestion. A missing horizon leaves availability
  explicitly unknown/unavailable, with no invented date. One compact
  scene table and one absorption-index caption; no calibrated concentration,
  emission rate or confirmed plume is claimed.
- **Evidence:** offline timing/cache/recipe tests; 927's three October scenes
  precede its spell. Optional Earthdata GeoTIFF downloads remain unimplemented.
- **Status:** active; imagery availability depends on the EE collection.

### D60 Monthly update runner (v5)
- **Choice:** `update.main(publish=False)` resumes pending segments before
  extending actual local nights through today−3 with `backfill.main`, then
  rebuilds and refreshes EMIT. Publication requires `publish=True`.
- **Immutable inputs:** `pipeline.persistent_jobs` writes compact gzip JSON
  (`expected_jobs_<tag>.json.gz`) with full expected job metadata atomically
  before processing. Compression preserves the exact original site lists,
  timestamps and public object keys. Existing uncompressed persisted snapshots
  remain readable. Resumes reuse the immutable set, preserving cached
  jobs even when NOAA stops listing them. Legacy granule manifests are unioned
  with the current listing; vanished legacy jobs remain expected/uncached.
  The runner also freezes `update_scope.json` before invoking a resumer, so
  rejection of a shrinking legacy manifest persists across later invocations.
  It rejects shrinking/changing expected sets or an empty expected
  listing before rebuilding or publication. Granule ownership in both pass and
  copy loaders goes to the latest **complete** manifest snapshot. An incomplete
  adjacent segment cannot erase an earlier complete recovery, including a pass
  on the previous local night outside its requested segment. Complete zero-copy
  and zero-pass replacements still supersede older positives/exposures. Older
  exports without completion flags predate partial-run acceptance and are
  treated as completed snapshots; explicit incomplete snapshots supply no rows.
- **Retention:** before any resume, inspect the oldest uncached job; legacy
  manifests use the segment start conservatively. Warn at 75 days, refuse at
  91 days until archives recover missing inputs. Fully cached jobs can be
  rebuilt after retention expires. `accept_missing` cannot bypass retention.
- **Missing inputs:** default tolerance is zero. Explicit `accept_missing=True`
  can acknowledge the exact recorded missing list after retries. Store it in
  `accepted_missing.json`; do not resume that unchanged accepted deficit every
  month. `missing_observations_<tag>.csv` records affected site-nights, which
  are exported as not observed (no negative exposure); available source
  detections remain positive evidence. Legacy jobs lacking site metadata make
  the whole segment unknown. Silent input disappearance is never accepted.
- **Boundary deficits:** acceptance verifies each missing granule separately.
  A saved job with known sites/time and no in-range local night is accepted
  with an empty exposure export (e.g. far-east UTC end+1), so it cannot block
  every subsequent month. A different missing job's exposure rows cannot cover
  up an unrecorded in-range deficit; legacy unknown scope gets no exception.
- **Conservative missing-night policy retained:** one missing granule still
  removes every exported pass for the affected site-night, including completed
  passes from other satellites. It can discard real observing opportunities,
  but cannot manufacture a clear miss. Source-copy detections remain positive
  evidence and can end a spell. The map may say "not observed" for that night;
  this deliberate abstention is broader than the missing granule alone.
- **Evidence:** shrinking-manifest, expired-resume, permanent-failure/next-month,
  empty-copy replacement, and three-segment full-build offline regressions.
- **Status:** active; no new VIIRS acquisition or publication in this change.

### D61 EMIT for every tier; flare-out explainer figures (v5)
- **Choice:** `emit.enrich` lists EMIT overpasses for High, Mid and Watch
  cards (D59 had High and Mid only). Watch spells are long and ongoing, so their
  ±14-night windows are wider; the listing, timing labels and chip rules are
  unchanged.
- **Result (4 Nov build):** the three Watch cards (3644, 3662, 6011) have no
  EMIT scene from early September to 4 November. NASA CMR lists 15, 17 and 14
  scenes at those sources since January 2025, the latest in July (3644, 3662)
  and June 2026 (6011), so the listing works and the gap is in acquisitions.
- **Explainer:** Figures 20–25 (`docs/explainer/make_flareout_figures.py`):
  one card night by night with its comparison flare, the tier rules, confidence
  by clear dark nights, the conditional heat bound, EMIT timing labels, and two
  pre-cutoff EMIT scenes drawn with the review recipe. Part 2 figures become
  26–28.
- **Confidence curve:** "this flare alone" is 1 − p₁·q^(k−1), the D58 Markov
  tail for k clear dark nights; "across the fleet" is 1 − min(1, F·p₁·q^(k−1)),
  the D58 conservative score, so High (score ≤ 0.1) is fleet confidence ≥ 90%.
  Bands span the 42 sources passing the bright rule at 4 November (p₁ 3.4–16%,
  F from the 4582/4583 spell, 1,348). High then needs 13 to 16 clear dark
  nights; with independent misses it would need 4. These are model
  sensitivities, not validated rates.
- **What a "miss" is:** no record shows a flare burning on a dark night. In the
  115 stable blocks, a clear miss followed a detected night on 1.7% of such
  nights (43/2,568). The 56 clear misses do not concentrate where detection is
  weakest: 42 had a near-nadir clear pass and 46 had ≥3 clear passes; after a
  detection the miss rate was 1.4% (27/1,902) on nights with ≥3 clear passes
  including near nadir, and 2.5–3.0% on single-pass nights (2/81, 1/42, 2/67).
  So the misses are consistent with a mix of detection failures, thin cloud
  the mask calls clear, turndowns and real sub-day outages. Counting them all
  as detection misses of a burning flare lowers fleet confidence (conservative
  for flagging). Per-source p₁ adds one hit and one miss, so the 3.4% low end of
  the 3.4–16% range is sources with zero misses (the added count alone). Explainer wording
  corrected from "a flare that kept burning … was missed" to this.
- **Real-world durations (cited, not modelled):** automatic pilot relight
  (minutes); Zaap-C unlit 17 days while venting, December 2021
  (Irakulis-Loitxate et al. 2022); 3–5% of US onshore flares unlit at a time
  (Plant et al. 2022); in EDF's 2020 repeat Permian surveys, over half of
  malfunctioning flares were unlit or partly lit on more than one day of a week.
  No offshore duration distribution is published.
- **Clear vs calendar nights:** 31 sites with median nightly heat ≥3 MW had a
  clear pass on a median 88% of the 88 nights (tenth percentile 26%).
- **EMIT example:** scene `20260621T173441_2617211_045` shows two parallel
  high-index bands; one joins 927's position to a point near Zaap-C, about 7 km
  east-southeast. Both narrow toward their east-southeast ends. The explainer
  describes the image only and assigns no source.
- **Status:** active.

### D62 Single dark nights and EMIT comparison scenes (v5)

- **Purpose and scope:** a second mode in Flare-outs groups qualifying clear
  dark nights into one list row per run. Dark spells stay the default;
  `#flare-outs/dark-nights` opens the mode, and a run id may follow that path.
  Same-source overlapping cards are linked without changing card rules or
  tiers. This screen identifies observations for review, never a cause.
- **Source and night:** reuse D58 `source_units`, `prepare_passes`,
  `prepare_copies` and `unit_history`. The discovery baseline ends 29 days
  after the first processed night. One raw cloud-mask 0/1 pass at SZA ≥97°
  qualifies a night. Any source-copy detection within 1.5 km, including cloudy
  and unassigned copies, vetoes darkness. For burning rates, a detection counts
  only within 1 km or when its nearest source unit is this one. Nights with
  only a neighbour's detections within 1.5 km are neither rate-lit nor dark;
  exclude them from both rate counts and show the excluded count. Heat uses
  clear groups within 1 km. D58 card selection is unchanged.
  **Round 4:** a clear dark night is not listed when another source unit within
  1.5 km had detections nearest to it that night. Detections from a neighbour
  can land just beyond 1.5 km (3780 on 1 Oct 2026: neighbour 3779-3781 at
  1,287 m, its detections 1,517–1,658 m from 3780 at 47–71 MW), so such a
  night cannot be attributed to either source. The count of skipped nights is
  published (`counts.skipped_neighbour_detected`). Nights lit only by a
  neighbour have their own strip state `N`.
  Member sites supply one union opportunity per satellite pass.
- **Run and row:** consecutive clear dark nights; cloud-only and unobserved
  nights do not split a run, but any detection does. Start/end are the first
  and last clear dark nights. A row shows dates, total clear dark nights,
  qualifying nights listed, and its best night's rate, heat with n and k clear
  passes. The detail shows bracket, phases, scenes, comparison scenes and card
  link once, then each qualifying night's own rate, heat, passes and provisional
  flag. Render 50 runs at a time with Show more. Use singular words for n=1.
- **Burning rate:** each selected night's inclusive ±14-night window is clipped
  to the processed record. Exclude that night and neighbour-only nights. All
  other qualified nights, including other clear dark nights in its run, count
  in the denominator. Lit-before evidence, brackets and comparison positives
  use detections attributed to this source by the rule above.
  Require ≥10 qualified nights and lit/qualified >0.5; exactly half fails.
  Require ≥1 lit qualified night before the run's first clear dark night inside
  that selected night's window. This removes not-yet-started sources and long
  runs with an inflated run-excluded baseline. Every listed run has a last-lit
  bracket. Keep lit/qualified counts inline. With fewer than 14 later processed
  nights, flag provisional; exactly 14 is complete.
- **Recent heat:** median nightly finite `heat_mw` over lit qualified nights in
  the same window. n counts measured nights, not groups or passes. Null heat
  with n=0 remains eligible. Heat filters: any (default), 0.5, 1, 3 MW; region
  filter also available. A run matches a heat filter if any qualifying night
  reaches it; the row still identifies its best night by review score.
- **Review order:** match heat to D58's `(lower, upper]` same-night re-detection
  classes. Keep the curve's p, detections/trials and the night's k clear passes.
  `miss_all=(1-p)^k`. Score = `wilson_lower_95(lit, qualified)*(1-miss_all)`,
  using the lower end of the two-sided Wilson 95% interval (z=1.959963984540054).
  Most surprising sorts highest score first, then higher recent heat, night,
  and run id. A run uses its best qualifying night's key. Unknown scores sort
  last. Brightest uses the highest measured recent heat among its qualifying
  nights; Most recent uses its latest qualifying night. Show both score parts
  in plain words with their n. Passes on one night are not independent: this
  is a review order, not a probability. It changes no selection or card tier.
- **Passes and bracket:** retain all union exposures of each qualifying night:
  satellite, UTC time, clear flag, view zone, detected flag. Near nadir is
  1008–2191; swath edge <640 or ≥2560; otherwise mid-swath; missing is unknown.
  Source-copy matches use satellite and granule or a time gap <10 seconds.
  Copy-only positives remain private enrichment evidence. The run bracket is
  last-lit before, first/last clear dark, next-lit after (null while still dark).
  Use `emit.scene_phase` for D59 phases. Daytime scenes next to single dark
  nights are usually boundary; no unobserved transition time is inferred.
- **EMIT listing and finality:** query each selected source and both products
  using stable full calendar-month caches covering the processed record. Reuse
  a month only when fetched ≥30 days after month end. Combine all remaining
  months in one query per product per source (at most two ordinary requests
  on a rebuild; pagination can add requests). Split successful results back
  into monthly snapshots, so months become final independently.
  Deduplicate acquisition records and restrict scenes to the processed record's
  noon-to-noon UTC interval. A failed monthly/product query makes the listing
  partial, or unavailable if all fail. Preserve offline snapshots without
  asserting complete coverage; never expose exception strings or credentials.
- **Nearby and comparison scenes:** nearby windows span two local nights before
  through two after the run, converted with longitude/15 and local noon bounds.
  Reference scene id, D59 phase and signed hours to the nearest clear dark pass
  anywhere in that run. A comparison requires the nearest passes strictly
  before and after to be source-positive, including cloudy copies. Exclude
  scenes in any listed run's ±2-night window at that source. Choose at most
  three remaining comparisons, closest to the runs' clear dark passes then id.
  Require measured EMIT coverage at the source for comparisons; unknown and
  no-data scenes cannot be comparisons. Label lit on both sides; do not infer
  continuous burning between passes.
- **EMIT source coverage (both modes):** the rendered index uses the continuum
  >0 validity mask. Store `source_pixel_mask`, `source_valid_pixels` within
  300 m, and `source_coverage` (`covered`, `no_data`, `unknown`). The source
  pixel must be valid and the nearby count positive to call it covered. Cache
  no-data as final, with the browse link and a plain missing-data message.
  Migrate legacy `ch4.png` caches offline using pixels inside the source ring;
  record that these counts are thumbnail pixels, not native raster pixels.
  Missing or unreadable PNGs stay unknown. Only covered scenes with both chips
  count as with-chips. Fold same-orbit tiles within 30 seconds into one
  acquisition, preferring source coverage and retaining adjacent-tile notes
  and browse links. Recipe `hitl-v4` renders masked pixels grey (#808080),
  distinct from the ≤1σ dark blue (#00131f); the shared caption explains this.
  Recipe `hitl-v5` (round 4) also fills pixels outside the scene footprint
  grey, in both the CH4 index and true colour, and both captions say "Grey: no
  data". Every published scene carries `source_coverage`; unrendered scenes
  say `unknown`. Recipe `hitl-v7` also masks EMIT fill in the
  true-colour bands before the stretch (−9999, or exact zeros in all three
  bands, measured at 7533); they showed as a black band along strip
  edges (7533, 30 Aug 2026) and now render grey. The fill mask is the one departure from
  `hitl_app._emit_truecolor`. (`hitl-v6` masked only −9999 and never shipped;
  the cache still falls back to v6–v3 chips, labelled as older renders.)
  Legacy covered chips remain reviewable offline and queue for the new recipe;
  their copied data pixels change from black to grey, preserving all stamps
  and the original cache. Legacy no-data tiles need no repeat render.
- **Chips, queue and bound:** use `emit.fetch_chips`, D59 recipe and successful
  cache before EE initialization, copying chips to `site_dir/img/emit/` even
  offline. Default bound: 60 new unique attempts per build. Nearby scenes in
  default run order precede controls. Persist attempts, last attempt, status,
  retry time and measured collection horizon in `emit_cache/darknight_attempts.json`.
  Reserve ceil(budget/2) for never-attempted scenes; spend the rest on due
  retries, allowing unused slots to go to new scenes. Failed renders back off
  1, 2, 4, 8, 16, then 32 days. Successful chips use no attempts. Scenes beyond
  the measured collection end say pending and use no budget, including a zero
  or negative render budget; reuse the saved horizon then, measuring if absent.
  Without a horizon, availability is unknown. Other scenes say queued, with
  a retry date when waiting after failure. An EE connection failure can retry
  next build and does not spend the render budget.
- **Compact payload:** one finite `darknights.json`, no detail files and no
  source pass histories. Private in-memory histories are available only during
  enrichment. Store each unit once; heat history is disjoint column blocks only
  within ±14 nights of listed runs (`start`, `heat[]`, state string). Codes:
  L detected/measured, H detected/unmeasured, D clear dark, U unknown. Every source scene
  appears once; runs reference `[scene_id, phase, hours]` and sources reference
  comparison ids. Store caption, definitions and curve once. Night pass arrays
  use satellite/view-zone indexes and seconds from that night's UTC midnight,
  preserving full times while reducing bytes. Target under about 1 MB on the
  current record; regress bytes per run and absence of source `passes`.
  Refresh through `build_viewer.main` and monthly `update.main`. Fetch only on
  first opening, with retry on failure; filter/sort the full run list locally.
- **Strip and phone:** chart the run plus 14 nights on either side. Dark is
  zero; lit without measured heat is a hollow diamond; unknown is a gap. Shade
  and label dark run. Cap at three times the best night's recent heat when
  measured, mark clipped points with triangles and retain exact hover values.
  Otherwise use the measured maximum. Use normal header wrapping, short UTC
  HH:MM:SS values, dates in the qualifying-night section, and compact headers
  for readable 390 px tables. Desktop keeps the existing two-column workspace.
- **Overlapping apertures:** link runs at different listed units within 1.5 km
  when they share a measured clear dark night; attach links in both directions.
  Say a neighbouring source within 1.5 km was also dark, with an aperture note.
- **Limits and evidence:** D61 observed 1.7% (43/2,568) over clear nights that
  followed a detected night, at flares seen on ≥80% of clear nights in 115
  stable 29-night blocks. The rate covers those eligible nights only. Most single
  dark nights at steady flares are unexplained. EMIT is daytime, hours from
  VIIRS; its CH4 index is not a concentration. Offline regressions cover these
  rules, cache progression and finality, compact payload, and monthly rebuild.
- **Networked build results:** pending; the main session will fill this line.
- **Status:** active; review material only.

### D63 NOAA's Azure copy for nights older than the AWS window (v5)
- **Choice:** `pipeline.main(source="azure")` reads the same NODD files from
  NOAA's Azure containers, `https://jpss.blob.core.windows.net/{snpp,noaa-20,noaa-21}`,
  under the same keys. `nodd.bucket_name` maps a satellite to an S3 bucket or
  an `azure:` container; `nodd.list_keys` pages S3 ListObjectsV2 or Azure List
  Blobs. The default source stays `aws`; `backfill.py` and `update.py` are
  unchanged.
- **Alternatives:** NASA LAADS L1B (`VNP02MOD`/`VJ102MOD`/`VJ202MOD` from
  2012/2018/2023; Earthdata login, a NetCDF reader and a calibration
  cross-check against IDPS); NOAA CLASS orders; AWS only (about 90 nights).
- **Why:** from 2026-09-14 NODD keeps M-band SDRs, GEO-TC and the cloud mask on
  AWS for 90 days. The Azure copy, listed in the same NODD registry entry
  under the same license, held these products back to 2023 on 2026-11-09.
- **Evidence (probes and runs 2026-11-09):**
  - *Coverage* (day folders per month): M-bands, GEO-TC and cloud mask from
    2023-06-22 (NOAA-20) and 2023-06 (SNPP); NOAA-21 M-bands and GEO-TC from
    2023-02-27, cloud mask from 2023-11-13. Months with missing days: NOAA-20
    cloud mask Dec 2023 (6/31 days), GEO-TC and mask Aug 2024 (13/31);
    NOAA-21 mask Dec 2024 (7/31); SNPP GEO-TC Oct 2023 (11/31), all three
    Jul 2024 (19/31), Aug 2024 (13/31 for GEO-TC and mask) and Jul 2026
    (22/31). M-band SDRs alone start 2022-08 for SNPP and NOAA-20, without
    GEO-TC before 2023-06. Sampled days held 1,012–1,013 granules per product.
  - *Same files:* listings of the 8 products × 3 satellites for UTC
    2026-08-08 to 11-08 found 2,126,617 files on AWS. Every one is on Azure
    with the same size and MD5 (0 mismatches). The 16,772 Azure-only files on
    days AWS still lists are all on UTC 08-10, the AWS deletion edge.
  - *Same results:* night 2026-09-20 rerun from Azure (1,292 granules,
    11.7 GB) reproduced all 2,584 per-granule cache files of `v5_backfill`
    byte for byte, and its 7,637 site-nights in every column. `granule.py`
    and `nightfire.py` are unchanged since the backfill commit `479736f`.
  - *Older nights* (one night each; 2026 GFW site list): 2023-09-15,
    2024-03-15, 2024-10-15 and 2025-06-15 completed 1,215–1,322 of
    1,215–1,322 granules in 8.6–9.5 min and 10.9–11.4 GB each. A one-night
    run reads two UTC days; a continuous run reads about 5.7 GB per night
    (D50).

    | Night (MYC year) | Flaring sites | Heat at flaring sites | Log-RH r vs MYC (n) | Median ratio | Per-pass detection, persistent (ours / MYC) |
    | --- | --- | --- | --- | --- | --- |
    | 2023-09-15 (2023) | 113 | 463 MW | 0.73 (101) | 0.83 | 0.65 (447/684) / 0.69 |
    | 2024-03-15 (2024) | 151 | 458 MW | 0.79 (138) | 0.97 | 0.67 (473/704) / 0.74 |
    | 2024-10-15 (2024) | 118 | 425 MW | 0.71 (109) | 0.93 | 0.58 (440/761) / 0.71 |
    | 2025-06-15 (2025) | 163 | 426 MW | 0.73 (146) | 0.75 | 0.53 (496/937) / 0.64 |
    | 2026-09-20 (2025) | 145 | 880 MW | 0.36 (125) | 0.88 | 0.48 (392/825) / 0.62 |

    The 87 full 2026 nights have 85–195 flaring sites (median 130) and
    238–963 MW (median 651, 5th percentile 441). At sites flaring on the
    old night and on ≥10 nights in 2026, old-night heat over the site's 2026
    median is 0.95–1.23 (n = 95–138) against 0.98 (n = 132) for 2026-09-20.
    The lower old-night sums come from sites larger in 2026: site 4267
    (25.18 N 52.88 E, Das Island, not in the upstream MYC catalogue) has a
    2026 median of 79 MW and 9–30 MW on the old nights; sites 4452 and 4513
    have MYC volumes of 0.000–0.001 bcm/yr in 2022–2025, were clear and dark
    on the old nights, and were lit on 77/82 clear nights in 2026.
- **Limits:** sites are the 2026 GFW list, so structures removed before 2026
  are missing and newer ones are included. NOAA-21 passes before 2023-11-13
  have no cloud mask and are never clear: on 2023-09-15 its 17,663 site
  passes held 297 detections, all counted under cloud. 2025-06-15 had 33
  not-observed site-nights, all at 59–72° N. Each old night is one sample.
- **Status:** active (v5); default source `aws`. The NASA L1B calibration
  comparison over the 2023–2026 overlap is not done.

### D64 Opt-in cold-run certificates with complete traffic accounting (v5)
- **Choice:** keep T0 traffic accounting always on; keep the existing processing
  defaults and 12 workers. T1/T2/T7 remain opt-in through `speedup.Options` /
  `pipeline.main(speedup={...})`, with separate output folders per option set.
  **Round 2 supersedes the round-1 audited footprint/index implementation and
  its cold-run conclusion:** remove per-object audits/ETag indexes entirely.
  T0, T4a/T4b and T5 retain their round-1 contracts. No new default is enabled.
  - **T0:** sidecars count response-body bytes and every HTTP attempt, including
    HEADs, errors, retries and empty jobs, plus CPU/wall time. Legacy cache
    traffic and killed-worker in-flight traffic remain explicitly unknown.
    Existing site/copy schemas stay unchanged.
  - **T1 v2:** skip when every site's solar bound excludes a night pixel within
    1.2 km, or every site is more than **25 km** outside the header G-Ring in a
    centroid gnomonic plane. Invalid geometry retains the job. The 25 km default
    supplies substantial margin despite zero wrong skips at 0/10/25/50/100 km.
    Never skip using the `Day` flag: two such August jobs produced rows. Reuse
    the opened GMTCO HDF5/range object and metadata blocks when retained.
  - **T7:** local daily TLE-AUX text, nearest epoch within three days, SGP4,
    WGS84 sites, GMST Earth rotation and the inertial VIIRS scan plane. Use the
    filename's decisecond start/end (midnight rollover included); header-only
    offline validation uses a nominal 84.5 s interval. A visible ±60° scan-angle
    and ±10 s temporal whitelist can skip before any download. Failures/stale
    TLEs retain jobs. Fit cross-track look-angle to native scan samples once:
    `raw = -3219.70057624 * angle_rad + 3152.42729942`; invert the fixed
    1:1/2:1/3:1/2:1/1:1 aggregation zones for columns. Rows use interval fraction
    minus five lines. Keep CMR job building; no orbit-only CMR replacement.
  - **T2 v2:** JRR chunk windows at predicted pixels with ±64 rows / ±96 columns.
    Require a nearest finite-M10 pixel within 1.2 km, an interior window hit
    with ±16-row bow-tie support, and no near tie. Widen once, then full GMTCO.
    Per-site 25 km footprint exclusions can certify misses. Main-session
    change (2026-11-10): a site predicted beyond the image by more than the
    margins is left out, and a nearest window pixel at ≥1.2 km is a certified
    miss, both only while the margins exceed the measured prediction error
    plus bow-tie overlap and radius (`speedup.margins_cover_error`: rows
    26.68 + 16 + 3, columns 14.05 + 3). Sol's round-2 rule sent every job with
    an off-image prediction to full GMTCO, which disabled windows on all three
    large Night samples. A job whose sites are all excluded returns no rows.
    Image-space thresholds require all eight G-Ring vertices and 32 divisions
    of every edge at SZA ≥96° (95° + **1° margin**), and reject an enclosed solar
    subpoint. Finite M10 supplies the image validity mask; load site/candidate
    halos for SZA, footprints, peaks, merges and centroids. Reuse JRR for clouds.
    Missing JRR (including early NOAA-21), uncertain night masks/layouts,
    prediction failures and pre-2026 products retain full reads. Historic fill
    differences prohibit claiming whole-array equality for that generation.
    Keep as experimental: the complete-job samples establish correctness and
    fallbacks, not a nightly sparse saving; all three large Night jobs contain
    unobserved sites and take the cheap full-read path.
  - **T3:** remove the draft flag/path. Non-detections still need site/cloud
    rows; no extra useful shortcut was demonstrated on the seven samples.
  - **T4a/T5:** keep optional tier-1 / NOAA-20 submission priorities, with stable
    chronological order otherwise. Tier 1 is the independently reproduced
    736-site lit union over 88 reference nights; no MYC additions. Satellite
    priority precedes site tier when combined. These options save no bytes.
  - **T4b:** explicit reduced product only on granules with no tier-1 site:
    omit non-detections and per-site cloud reads; retained detections have
    `cloud_mask=-1`, `cloud_clear_neighbors=0`. Preserve copy records and peak
    cloud evidence. Suppress normal aggregation; write the reduced table.
    Exposure and clear/cloudy denominators cannot be computed from that product.
  - **T6/harness:** control/best/control on Aug 14, Sep 20, Oct 25; historic
    control/best on Jun 20, 2024, using that control as its reference; then best
    at 24/32 workers on Sep 20 (12 already measured). Best means T1+T7+T2,
    not a measured winner. No audits or reused HTTP/science caches. Rebuild
    boundary site unions using adjacent-day CMR slices, without extra SDR
    listings or selecting sites from reference values. A 9,000 s cumulative
    supervisor deadline terminates the process group and marks in-flight
    traffic unknown. Stop immediately on identity/accounting failure; persist
    differences and fallback/skip reasons. Reject partial processing caches.
- **Alternatives:** preserve round-1 per-object full-array proofs; use `Day`
  skips; narrow orbit margins; trust unbounded local hits; use image thresholds
  across the terminator; silently normalize historic fills; assume sparse
  stress subsets establish whole-job/network savings; change production defaults.
- **Why:** audits consume the bytes a cold run should avoid. These cheap
  certificates use real-data margins and full-read fallback. Orbit prediction
  supplies useful windows and avoids header requests, even though its job
  exclusions add no jobs beyond T1 on the supplied nights. Conservative misses
  can defeat T2; preflight avoids that extra traffic where predictions already
  show the problem. Live timing and full-band/reference identity remain gates.
- **Evidence (2026-11-09/10, offline):** all three nights have 3,896 job headers,
  2,460 rows-producing jobs and 170,715 reference site rows. T1 25 km plus solar
  skips 414/417/463 jobs (1,294 total), **zero wrong skips**. Across the five
  tested buffers, wrong job skips remain zero; per-site 25 km exclusions also
  miss zero reference rows. T7 drops 333/361/397 jobs (1,091 total), zero wrong
  jobs or reference-site exclusions. Including solar gives 1,135 zero-download
  jobs. Absolute row errors (n=170,715): median 3.86, p95 10.74, p99 14.47,
  max 26.68; columns: 2.26, 11.31, 12.38, 14.05. Cross-track fit n=16,800
  sampled real pixels across all seven granules. Full float32 JRR/GMTCO equality
  holds for six 2026 arrays of 2,457,600 elements per coordinate; 2024 differs
  at 51,200 fill pixels (`-999.3` vs `-999.0`), with zero valid-coordinate
  differences. Finite-M10/invalid-GEO violations: 0/7 granules. Night certificates
  accept 3/7 (true minima 126.72–128.56°), reject all Both/Day samples (true
  minima 83.87–89.69°); edge minima are conservative within 0.030° on all seven.
  Complete-job M10/GEO/cloud sample outputs agree with control for 7/7 granules
  (14 CSVs excluding only `bytes_granule`), 4,384 site rows; reference nearest
  pixels agree on all 4,382 available rows. Confirmation-band data were not
  supplied, so this is not a real full-fit/reference-file equivalence claim.
  Separate sparse stress checks and synthetic detection/copy tests cover that
  path. Adjacent CMR reconstruction matches all 3,896 reference site counts
  and omits zero reference rows. Frozen HEAD/default synthetic CSV bytes agree
  on 2/2 files including `bytes_granule`. No live HTTP, new live benchmark,
  commit, push, publication or backfill was run. See `outputs/speedup/REPORT.md`
  and machine-readable evidence for tests and measurements. Offline suite: 274
  tests; ruff check and formatting of changed Python files pass. The five
  pre-existing unrelated formatting failures remain untouched. Baseline 220-night
  projection remains roughly 14–15 h (D50/spec); optimized projection awaits live.
- **Live evidence (main session, 2026-11-10; Azure; `outputs/speedup/cold_r2b/`):**
  every run reproduced its reference: all per-granule site CSVs equal except
  `bytes_granule`, all copy CSVs raw-identical, 0 wrong skips (2,366–2,688
  files per run). Single-night runs read two UTC days.

  | Night | Run | Workers | Download | Requests | Wall |
  | --- | --- | --- | --- | --- | --- |
  | 2026-08-14 | control / best / control | 12 | 15.46 / 5.94 / 15.46 GB | 23,252 / 34,767 / 23,252 | 8.4 / 6.0 / 8.3 min |
  | 2026-09-20 | control / best / control | 12 | 15.90 / 5.96 / 15.90 GB | 24,241 / 36,525 / 24,241 | 8.3 / 6.4 / 8.3 min |
  | 2026-09-20 | best | 24 / 32 | 5.96 GB | 36,525 | 3.5 / 3.0 min |
  | 2026-10-25 | control / best / control | 12 | 15.47 / 4.82 / 15.47 GB | 23,387 / 35,237 / 23,387 | 8.5 / 6.3 / 9.3 min |
  | 2024-06-20 | control / best | 12 | 14.14 / 10.02 GB | 20,907 / 18,800 | 7.9 / 5.8 min |

  Best mode cuts bytes 2.6–3.2× on 2026 nights but adds ~50% requests; at 12
  workers it is latency-bound (Azure first byte 0.44 s; 13–17 MB/s). At 32
  workers it reaches 33 MB/s, near the 34–37 MB/s link, 2.8× faster than
  control (3.0 vs 8.3 min). Job paths on 2026-09-20 (n=1,292): windowed 847
  (5.70 GB), orbit skip 363, header skip 49, solar skip 5, night-mask fallback
  28 (0.26 GB); 68,155 far-outside site predictions left out and 6,343
  certified misses. 2024-06-20 saves only the skips: 777 jobs use full GMTCO
  (pre-2026 fill sentinels), 7 lack JRR.
- **Main-session operational changes:** `pipeline.build_jobs` lists days in
  parallel (16 threads, merged in day order; 3 nights in 16 s), since serial
  listing would take ~40 min before a 220-night run; `backfill.main` passes
  `source` and `speedup` through.
- **Status:** T0 active. T1 v2 + T7 + T2 v2 ("best") at 32 workers passed live
  identity on four nights and runs the 2026 backlog (01-01..08-08,
  `outputs/v5_backlog_2026/`); still opt-in, not the default. T4/T5 optional
  orderings; T4b reduced product experimental; T3 removed.

### D65 2026 backlog from Azure with the D64 fast mode (v5)
- **Choice:** local nights 2026-01-01..08-08 (220) from NOAA's Azure copy
  (D63) with T1 v2 + T7 + T2 v2 at 32 workers (D64), through `backfill.py`
  into `outputs/v5_backlog_2026/` (run script `outputs/backlog_2026/run_backlog.py`,
  orbit files `outputs/backlog_2026/tle/`). `pipeline._run_pass` now keeps at
  most 4 queued jobs per worker (`IN_FLIGHT_PER_WORKER`).
- **Alternatives:** the current method at 12 workers (projected 14–17 h:
  D50 3.9 min/night; live single-night controls 8.3–9.3 min per two UTC
  days); a cloud machine next to the data in Amsterdam (not set up).
- **Why:** the user asked for the backlog once the fast mode passed live
  identity (D64) and for its timing.
- **Evidence (2026-11-10):**
  - *Deadlock:* the first attempt (06:48 UTC) built 134,364 jobs in 6.5 min,
    then hung for 2 h with no results: CPython 3.10's `ProcessPoolExecutor`
    blocked the main thread in `os.write` on its wake-up pipe after all jobs
    were submitted at once (all 32 workers idle). Earlier runs submitted at
    most 36,875. With bounded submission, 80,000 jobs passed through 32
    workers in 15 s; test `test_bounded_completed_limits_queue_and_keeps_order`.
  - *Run:* restarted 08:51:30 UTC from the frozen job list, done 13:59:48 UTC:
    **308.3 min** (processing 292.3 min, then aggregation and MYC
    validation), 134,364/134,364 granules, 0 errors, 586.7 GB (2.67 GB per
    night; 33.4 MB/s, the link limit). Throughput fell from 503 to 428
    granules/min as summer nights brought more night-mask fallbacks.
  - *Results:* 1,680,140 site-nights (7,637 × 220): clear, no flare 56.9%;
    cloudy 40.7%; flaring 1.61%; flaring under cloud 0.73%; not observed
    0.10% (1,711, mostly May–July at high latitude). Flaring sites per night
    median 124 (range 63–227) against 130 (85–195) on the 87 later nights;
    summed heat median 552 MW (140–1,015) against 651 MW (238–963). Against
    MYC25Q4 2025: log-RH r = 0.75 (n = 503), median ratio 0.67; per-pass
    detection at persistent sites, like-for-like, 0.58 (86,005/155,811)
    against 0.63; persistent sites detected at least once 96.6% (344/356).
  - *March–April:* Persian Gulf heat fell from 364–380 MW per night (Jan–Feb)
    to 125 (Mar) and 174 (Apr), then 344 (May) and 601 (Jun); its lit share
    of clear site-nights fell from 0.034–0.037 to 0.020. NOAA-20/21 detection
    counts stay within 4,000–4,600 per month, so it is not a single-sensor
    gap. SNPP is low in March and July, matching its missing Azure days
    (25/31, 22/31); NASA CMR lists no SNPP night footprints on 07-14..16 and
    08-04..05. The cause of the Gulf drop is not established.
- **Status:** code in `26424c8` on `nightly-flaring`. Results copied to the
  production `outputs/v5_backlog_2026/` (403,103 files; run folder with TLEs in
  `outputs/backlog_2026/`, test evidence in `outputs/speedup/`).
  `build_viewer.find_segments` joins it with `v5_backfill` and `v5_30n` into
  2026-01-01..11-04 (308 nights); the website is not rebuilt or published.

### D66 Long-record time axes and content-addressed yearly nights (v5)

- **Choice:** follow [the design](long_record_ux.md) with the round-3
  corrections. All bars, brushes and pointer inversion use elapsed nights;
  partial bins span only their nights. Step, range and pass-window markers
  stay centred on their time extent and have a 2 px minimum width. Weekly
  and monthly focus axes use `d3.utcMonth.every(k)`, the smallest fitting
  stride in 1, 2, 3, 6 or 12 months, or 2, 5 or 10 years, with at least
  6 px between labels (focus strip, site record, dark-run histogram and spell
  timeline). The first kept tick in each calendar year names the year. The bin caption has a separate line.
- **Controls and periods:** one Mean (range) / Night / Week / Month control,
  with conditional Quarter and Year, fits one deliberate row in the 400 px
  panel. Year buttons retain focus when activated. Whole calendar year,
  quarter, month and week ranges shift by their unit, preserving leap-year
  and month boundaries; other ranges, including units clipped by the record's
  first or last night, shift by their night count. Play picks
  Night through 92 nights, Week through 730, otherwise Month, and retains the
  chosen unit. Strongest rows show step figures; consistency and clear-view
  orders show range figures. Partial steps name the night count once.
- **Aggregation:** retain each site's mean over finite clear-sky values,
  including clear zeros and existing finite cloudy-lit values. Network and
  regional bars sum site means. Raw-export pandas reductions give monthly
  totals **1.196371–1.886898×** the mean of their nightly totals (**n=11**,
  January–November 2026 including partial November). The excess comes from
  sites with a value in the month but none on a given night. The timeline
  information, tooltip and Reading the app explain this width effect.
  For the Clear this unit tile and bin tooltip, a clear night has status F
  or C. Both count sites with at least one such night, matching the previous
  single-night display. Oct 19–25 counts **7,254**; Q1 2026 counts **7,619**.
  Two additional sites in that week have only cloudy-lit finite zeros; those
  values remain in the heat mean. The Flaring sites context retains its
  disjoint clear-sky/cloudy-lit denominator.
- **Data delivery:** plain JSON network totals for all five units and regional
  totals for week/month/quarter/year remain embedded. Sparse status/heat years
  use `data/nights-YYYY.<hash8>.json` in `meta.chunks`; clear-pass digits for
  every site use independent `data/clearpasses-YYYY.<hash8>.json` files in
  `meta.clearpasses`. Each kind depends only on that year's nights. Empty
  site-years are omitted and restored empty. The map renders after status and
  heat load; the clear-pass fetch starts after two animation frames. Until
  then Most clear views is disabled with a short loading label and the site
  fact shows “…”; other controls remain usable. The real 2026 files are
  identical across all three builds.
  Publication retains both kinds of files named by the preceding index for
  one build. A 404 refreshes the index without cache once under a guard for that file kind;
  successful loading clears the guard. Retry after a guarded 404 refreshes
  the index and reads its current file names. File/HTTP failures name the
  file and status; serve-over-HTTP advice appears only for `file:`.
- **Computation and accessibility:** only ever-lit sites allocate Float64
  heat arrays, preserving single-night display precision. Seven cumulative
  statistics at 32-night boundaries cover heat sum, finite-value nights,
  lit, observed, cloudy-lit, passes and F/C nights. Deferred passes fill only
  their own prefix column. Cached site panels keep calendar/record/pass nodes
  stable, including one empty-state paragraph for never-flaring sites.
  Calendar years include the newest four plus the range end, current step
  and pass window. Grid cells have row parents and one roving stop on a
  rendered night, initially the last pass-window night. Only consumed cell
  keys stop propagation; expansion buttons keep native Enter/Space and Esc
  reaches the page. Dark-run dates clamp to the record and inputs, brush and
  count reset together on neighbour navigation. Spell lanes lead with site
  IDs on phones. Spell charts permit touch scrolling; histogram touch drags
  filter horizontally while vertical swipes scroll the page.
- **Other choices:** all small charts use the calendar bin ladder and name
  their units. Sparklines cover the selected range; record and region charts
  cover the whole record. Night steps show 14 centred pass nights; longer
  units show their own nights through 21, otherwise their last 14. Playback
  writes the URL only on Pause. Incoming hashes restore before serialization,
  close omitted sites and clamp dates. Twelve months is hidden when equal to
  All. Debug arrays exist only under `?debug` and have capped samples.
  Synthetic builds read the current real build, retain real 2026 and seed
  per-site/year factors and start/stop changes in older years. Evidence,
  screening and pass files remain unchanged. EMIT listing is not rerun.
- **Alternatives:** a scroll-only nightly strip loses context; a pooled strip
  loses night access; strongest-status bins overstate persistence; fully
  embedded site nights invalidate the whole payload. Progressive loading of
  heat/status, change maps and small multiples remain deferred.
- **Why:** preserve individual-night evidence and responsive comparison across
  308, 1,500 and 5,000 nights, while keeping the initial transfer below round 1.
- **Evidence correction:** round 1 checked headline tiles without their
  contexts, accepted an inert overview drag and timed playback without an
  open site. Round 2 established parity for range means (All and Last 7/30/90)
  and the listed views; its blanket parity/calendar claims were too broad.
  Council found hidden older-year tab stops, swallowed Escape, irregular ticks,
  duplicate pass messages and 726 single-night rounding mismatches. The
  round-3 checks below exercise those failing paths directly. No claim here
  establishes long-record screening-method readiness, Safari or screen-reader
  behavior.

- **Evidence, round 3:** system Chrome at 1440/1024/390 px on all three
  builds, 2026-11-10. **286 pytest tests** and the pinned Ruff/pre-commit hook
  pass. All **43 browser checks per build** pass with zero runtime errors;
  **81 screenshots** are saved under `outputs/ui_checks/`. The tick matrix
  covers **87** combinations of build, width and All/12 months/Last 90/every
  Year. **40,817 single-night values: 0 displayed mismatches**; six complete
  nightly views match tiles/contexts, top-25 names/values and regional figures.
  Range-mean parity covers All and Last 7/30/90; step row counts follow their
  chosen period. All **7,637** raw histories, **40** file hashes and **8,109**
  network bins validate; maximum independent network/regional error is
  **0.000000499915 MW**. Real Oct 19–25/Q1 clear counts are 7,254/7,619.
  Older-range calendars, native expansion keys/Escape, site-0 empty states,
  date synchronization, touch swipes, focus retention, calendar shifts and
  both file-kind 404/Retry guards are asserted. The real Git retention test
  uses three local fixture publications and expires both kinds after one build.

| 5,000-night measure | Measured | Budget |
| --- | --- | --- |
| HTML growth / 1,000 nights | 82.48 KB | ≤150 KB |
| Header + overview first paint | 260.4 ms | ≤1,000 ms |
| Interactive after night files | 934.9 ms | ≤4,000 ms |
| All range statistics, median | 4.5 ms | ≤30 ms |
| Range refresh through paint, median, site closed | 62.4 ms | ≤150 ms |
| Range through paint, max, site closed | 66.7 ms | ≤150 ms |
| Range through paint, max, site open / all years | 66.9 ms | ≤150 ms |
| Playback median / p95, site closed | 33.3 / 33.4 ms | ≤42 / ≤67 ms |
| Playback median / p95, site open / all years | 33.3 / 33.4 ms | ≤42 / ≤67 ms |
| JS heap + backing storage after GC | 208.9 MB | ≤350 MB |

  The 1,500-night interactive time is **467.6 ms** (≤2,000 ms).
  Open-site timing asserts all **5,000** calendar cells. Range-through-paint
  includes two animation frames; playback samples 2.5 seconds per case.
  Heap includes backing stores after deferred loading. URL writes are zero
  during playback and one on Pause.

| Build | Nights / sites | Final index bytes | Night files: count / raw bytes | Deferred clear-pass files: count / raw bytes | Full build time / peak RSS |
| --- | --- | --- | --- | --- | --- |
| site | 308 / 7,637 | 3,302,196 | 1 / 2,763,845 | 1 / 2,427,501 | 3.05 s / 690.8 MB |
| site_long_1500 | 1,500 / 7,637 | 3,399,277 | 5 / 13,121,009 | 5 / 11,832,024 | 9.64 s / 290.7 MB |
| site_long_5000 | 5,000 / 7,637 | 3,687,949 | 14 / 43,410,904 | 14 / 39,239,270 | 32.83 s / 408.0 MB |

  Real index before: **6,443,540 bytes**. Data-build timings/RSS precede the
  final template/explainer-only refresh; sizes reflect the final artifacts.

| Build | Index gzip bytes | Night gzip bytes | Critical gzip bytes | Deferred clear-pass gzip bytes | 10 Mbps data-ready | 50 Mbps data-ready |
| --- | --- | --- | --- | --- | --- | --- |
| site_before | 1,147,137 | 0 | 1,147,137 | 0 | 1.259 s | 0.513 s |
| site | 635,406 | 409,765 | 1,045,171 | 534,672 | 1.205 s | 0.540 s |
| site_long_1500 | 663,210 | 1,674,457 | 2,337,667 | 2,267,238 | 2.375 s | 0.940 s |
| site_long_5000 | 742,467 | 5,509,108 | 6,251,575 | 7,546,766 | 5.920 s | 1.913 s |

  Transfer bytes above are exact served Node gzip level-6 bodies; cold
  data-ready times are medians of **n=3** fresh Chrome contexts at each speed,
  with **40 ms** latency. Critical transfer is index plus status/heat years;
  the separate column shows the later clear-pass transfer. System/Python
  gzip-6 critical bytes are 1,166,336 before, 1,062,388 real,
  2,387,053 at 1,500 nights and 6,401,823 at 5,000. The real
  initial payload is below the original 1.17 MB baseline and round-1 target.
  Deferral reduces time to a usable map; eventual bandwidth includes both kinds.
  Source data/evidence remain read-only; no screening or network listing changed.
  Full evidence: `outputs/review/round3_implementation_report.md`,
  `outputs/ui_checks/time_axes_report.json`,
  `outputs/review/round3_data_checks.json`, and
  `outputs/review/round3_transfer.json`.

- **Post-review fixes (main session, after Review 8 round 3):**
  - Session storage is read through a guard that treats a denied store as
    "no guard" and reloads only when the guard can be stored. Before, a
    browser blocking site data never finished loading (O3-1).
  - Calendar arrows follow the layout: ←/→ move a week (one column), ↑/↓ a
    day (one row). The year's SVG has `role="none"`, so the rows sit directly
    under their row group (O3-2).
  - Time-axis labels keep at least 6 px between them and may step by 2, 5
    or 10 years; before, labels touched or overlapped by up to 0.6 px at
    390 px (O3-3).
  - Overview year labels use a stride counted back from the newest year,
    widened until no two collide. Before, 2026 displaced 2025 at 390 px on
    5,000 nights (O3-7, M3-1).
  - Year files load with the browser's default cache: their names change
    with their content (O3-10).
  - After "Show earlier years", focus returns to the calendar cell (O3-5).
  - The flaring tile's (i) explains why its clear-sky count can be lower
    than the count of sites with any clear night (O3-6).
- **Open:** ‹ › over a unit clipped by the record's edge shift by night
  count, so the step is not reversible there (O3-4); the calendar does not
  mark hidden years between the newest four and an older selected year
  (O3-5); a failed clear-pass fetch leaves "…" in place (O3-8); at 5,000
  nights the deferred clear-pass files are 7.67 MB gzip (O3-12).
- **Status:** Review 8 GO from both reviewers in round 3; published with the
  post-review fixes above.

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

Ten local-night detection records represent five unique split components. Their non-isolated-cloud share is 80% (8/10); v4 had nine components.
Four local-night records at four sites have multiple sources; their swath-edge
share is 0% (0/4), against 29.8% (111/372) of all detections. Edge means peak
column <640 or ≥2560. Local-night records represent 32 merged copies. Across all processed granules, 1,046 copies form 954 groups, with
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
