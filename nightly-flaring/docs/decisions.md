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
- **Status:** active (v4). Review 3 recommends changes (N1, M7/M8 guard; see reviews.md).

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
- **Status:** active (v4). Review 3 recommends changes (N6, estimator label; see reviews.md).

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
- **Status:** active (v4). Review 3 recommends changes (N5 saddle test; see reviews.md).

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
- **Status:** merge distance superseded by D32 (v4); summing distinct sources
  remains active. Review 3 recommends changes (N2, N3, N7; see reviews.md).

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
- **Status:** active (v4). Review 3 recommends changes (N8; see reviews.md).

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
- **Status:** active (v4). Review 3 recommends changes (copy-level table; see reviews.md).

### D30 Empty coverage (v4)
- **Choice:** Skip a satellite when CMR produces no site-bearing slices for the
  requested dates.
- **Alternatives:** Fail on missing slice_start; fabricate jobs.
- **Why:** The requested November regression job build encountered empty SNPP coverage.
  Returning no jobs for that satellite preserves the normal NOAA-20 job selection.
- **Evidence:** Nine requested regression job builds cover ten site cases; empty SNPP
  coverage previously stopped a November build (n=1 observed failure). All ten
  comparisons now complete. The network-free all-empty-coverage test passes.
- **Status:** active (v4). Review 3 recommends changes (logging; see reviews.md).

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
- **Status:** active (v4).

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
- **Status:** active (v4). Two distinct flares less than one pixel apart in
  adjacent scans now merge; they are not resolvable at M-band resolution. Review 3 recommends changes (N2, N4; see reviews.md).

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
