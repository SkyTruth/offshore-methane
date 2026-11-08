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
- **Status:** active; splitting is fix 3 of review 2.

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
- **Status:** under revision for v4.

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
- **Status:** under revision for v4.

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
- **Status:** under revision for v4.

### D18 Nightly value and status (v3)
- **Choice:** `rh_mw` = mean RH over strict-clear passes with clear
  non-detections as 0 (form of rs2021 Eq. 3 and MYC25Q4). Status precedence
  `flaring` > `flaring_cloud` > `clear_no_flare` > `cloudy` > `not_observed`;
  `flaring_cloud` = a detection under a non-isolated cloud flag and no clear
  detection that night.
- **Evidence:** review 2 B10: 218 of 1,634 `flaring_cloud` nights also have a
  clear non-detection (`rh_mw` = 0); README step 6 said `cloudy`.
- **Status:** under revision for v4 (precedence and documentation).

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
- **Status:** detection-frequency definition under revision for v4; matching
  is fix 4 of review 2.

### D21 Execution and caching (v1)
- **Choice:** `ProcessPoolExecutor` (h5py serialises threads); one CSV per
  granule under `overpasses_<PIPELINE_VERSION>`; up to 3 passes retry failed
  granules.
- **Evidence:** v3, 30 nights: 19,443 granules, 167 GB, 105 min with 12
  workers on a laptop, about 26 MB/s.
- **Status:** active.
