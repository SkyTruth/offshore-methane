# Reviews

Each review is a two-model council: GPT-6.1 Sol and Claude Opus 5.5 get the
same brief (code, papers in the scratch text extractions, run outputs) and
report independently. Counts below are the reviewers' measurements unless
marked *checked*, which means the main session re-derived them from the
outputs. Choices made in response are recorded in [decisions.md](decisions.md).

## Review 3: v4, development night and spot granules (2026-11-08)

Code reviewed: commit `54dbe6e`. No 30-night v4 run exists; evidence is the
2026-10-29 development night, the nine regression granules, and live v4 runs
on spot granules (Sol 6, Opus 10). Both reviewers ran the 51 unit tests (all
pass). Scratch: `/private/tmp/nf-review4-gpt61/`, `/private/tmp/nf-review4-opus55/`.

### Verdicts

- **Sol (7/10):** no-go for the 30-night rerun until band elimination selects
  guard-valid alternatives (N1) and copy grouping cannot join two peaks from
  one scan (N2). Then a diagnostic 30-night rerun; backfill waits for
  attribution and RH-tail checks.
- **Opus:** v4 fixes the fit failures behind v3's largest values and most
  cross-site bow-tie double counting. Go for the rerun after two changes to
  per-granule output (centroid site assignment for merged copies, N3, and a
  per-granule copy table), because any later change there needs another full
  run. No-go for the backfill until the rerun passes the checks listed below.

### Review-2 bugs in v4

| Bug | Sol | Opus |
| --- | --- | --- |
| B1 M7+M8-only fits | Fixed (site 3802 3,126.9 → 124.3 MW) | Fixed; all 8 v3 records resolve; M10+M7+M8 final sets still possible |
| B2 RH75% threshold | Partly: fixed case, but a modified estimator (en2025 uses local RHmax) | Fixed in 6/6 cases checked, by the fit guards; cap excluded 0/1,114 dev pixels and 0 in 10 spot granules |
| B3 Two-band fits | Fixed as a guardrail | Partly: reduced χ² still ≡ 0; 17/820 dev detections `pinned:two_band_range` |
| B4 Low temperature | Partly: pinned, not excluded as in the catalog | Fixed: 0 fits below 1,200 K, 6/820 `pinned:low_t` |
| B5 Copies at two sites | Partly (N2; strongest copy sets the site) | Largely fixed: 18 v3 dev detections that vanish are copies merged into a neighbour (293.7 MW); N3, N4 remain |
| B6 Connected flares | Fixed for resolved maxima | Fixed, side effect N5 |
| B7 Discarded sources | Partly | Fixed: 8/824 dev records have two sources |
| B8 Peak location | Fixed | Fixed |
| B9 Dedupe | Fixed | Fixed |
| B10 Status docs | Fixed | Fixed |
| B11 Greedy matching | Not fixed | Not fixed |
| B12 Frequency definitions | Partly (N8) | Partly (N8) |

### New findings

| ID | Finding | Location | Evidence | Found by | Blocking |
| --- | --- | --- | --- | --- | --- |
| N1 | Leave-one-out ranks trials by χ² before the temperature/ESF guards, so it can keep a subset that then fails over one that passes | `nightfire.py:160–165` | Bands M10, M7, M11, M12 = 3.617259, 3.569423, 10.438807, 6.260087, noise 0.001: dropping M7 gives χ² 7.85 at 1,077 K and is chosen, then pinned; dropping M10 gives χ² 8.28 at 1,697 K, valid. *Checked.* | Sol | Rerun |
| N2 | Transitive copy merging can join two peaks from the same scan | `granule.py:240–256` | Two scan-0 peaks 1.6 km apart and a scan-1 peak 0.8 km from each (1 km pixels) become one 20 MW source; the second scan-0 source is lost. Production incidence unmeasured. *Checked.* | Sol | Rerun |
| N3 | A merged copy group takes the stronger copy's location, and so its site | `granule.py:259–261` | NOAA-21 `d20261030_t2118195`: copies 97.5 MW (161 m from site 4279) and 127.9 MW (860 m from 4281) merge and go to 4281; their centroid is 472 m from 4279, and MYC `s_9421` (0.727 BCM/yr) is 172 m from 4279. A pass 1.7 h later sends the flare to 4279. An RH-weighted centroid changes the site for 22% (8/37) of merged pairs in 10 spot granules; in the 2 of those 8 with one MYC flare nearby, MYC sides with the centroid. | Opus | Rerun |
| N4 | Copies 1.0–1.35 pixel apart stay separate and count at two sites | `granule.py:251` | 45 adjacent-scan pairs with \|Δcol\| ≤ 1 and Δrow 4–10: 41 are ≤ 0.90 pixel apart with RH ratio ≤ 2.1 (one at 4.0); 1.06 pixel (sites 4197 + 4202, ratio 1.33, one MYC flare `m_208`) and 1.32 pixel (971 + 957, ratio 1.82) stay separate; pairs at 1.45 and 1.79 pixel have RH ratios 121 and 89. | Opus | Cheap to add |
| N5 | Peak splitting has no prominence (saddle) test | `granule.py:148–159` | NOAA-21 `d20261030_t0748551` (cloud mask 3): one 48-pixel component with 0.19–0.56 MW peaks becomes 3 sources at 3 sites, plus 4 in neighbouring components. 65% (17/26) of split-tagged dev detections are cloud-flagged vs 29% (236/824) of all. Clear-sky splits also recover real flares (site 984 12.8 MW vs MYC `m_82` ≈ 13.6 MW). | Opus | Cheap to add |
| N6 | RH75% sums pixels anywhere in the partition, not only those adjacent to the peak (en2025 merges adjacent pixels) | `granule.py:202` | Site 981: 1.64 MW from 9 pixels, 8.9× its 0.19 MW peak. | Opus | Cheap to add |
| N7 | A merged group keeps the larger copy's RH | `granule.py:259` | Median RH75% ratio between merged copies 1.25 (n = 41); swath-edge merges read about 11% above the copies' mean (inferred). | Opus | Cheap to add |
| N8 | MYC-centred frequency selects GFW sites within 2 km but counts their detections wherever the peak is | `validate_myc25.py:84–86` | 7 dev rows have peaks beyond 2 km of the MYC flare, e.g. `m_92`/site 989 at 2,574 m, `s_5353`/979 at 2,736 m (Sol). In dense fields a nearby site's detection can be a different flare (Opus). Detection-based clearing differs from the MYC README's cloud 0/1 rule. | Both | Validation |
| N9 | Minor | `pipeline.py:65–66`, `granule.py:307–313`, `nightfire.py:150, 158` | Empty satellite skipped silently; `rh_mw` sums sources while `temp_k`/`area_m2` describe the strongest; `n_clusters` changed meaning; two-band χ² ≡ 0; M10+M7+M8 final sets allowed (v3: 19 such fits at 1,234–1,321 K, 479 MW). | Both | No |

### Decisions D22–D32

| D | Sol | Opus |
| --- | --- | --- |
| D22 fit guards | Change: rank only guard-valid trials (N1) | Agree; forbid final sets whose non-M10/M11 bands are only M7/M8. 1,200 K floor carries the volume effect (RH vs 1,810 K pin: 5.1× at 1,000 K, 2.4× at 1,200 K, 0.91× at 2,500 K) |
| D23 RH75% | Change: label as a modified RH75 estimator; no cap in en2025 | Agree; add adjacency (N6); cap inert so far |
| D24 split | Agree provisionally | Change: saddle test (saddle M10 ≤ 0.5 × lower peak), record prominence |
| D25 merge and sum | Change: no unconstrained transitive unions; export constituent sources | Agree on merge-then-assign and summing; use RH-weighted centroid (N3) and mean RH (N7) |
| D26 dedupe | Agree | Agree |
| D27 status | Agree | Agree |
| D28 frequency | Change: peak-radius leakage (N8); call detection-based clearing a sensitivity definition | Agree as an added metric; quote after optimal matching (B11) |
| D29 outputs | Agree; add source provenance and rejected-fit temperatures | Agree; add a copy-level table |
| D30 empty coverage | Agree; report skipped coverage | Agree; print skipped satellites and days |
| D31 dev scope | Agree | Agree; the dev night is unrepresentative (local night +0.7% strict-clear RH; all 820 detections in those granules −8.8%) |
| D32 one-pixel merge | Change: add group constraints; deviation from rs2019 | Agree over half-pixel; extend to < 1.5 pixel when \|Δcol\| ≤ 1, Δrow 4–10 and RH ratio < 3 |

### Spot checks (v4 code, live)

| Case | v3 | v4 |
| --- | --- | --- |
| NOAA-20 Oct 10, site 3802 | 3,126.9 MW, 916 K | 124.3 MW, 1,699 K |
| NOAA-20 Oct 21, site 4500 | 595.0 MW (2 clusters) | 39.1 MW; copies 39.1/35.0 MW merged at 0.30 pixel |
| NOAA-20 Nov 5, site 4279 | 645.1 MW | 21.9 MW |
| NOAA-20 Oct 22, site 4452 | 149.8 MW, 1,234 K, M10+M7+M8 | 54.1 MW, 1,752 K, `eliminated:M7` |
| NOAA-21 Oct 13, sites 3801 + 3804 | 165.7 + 150.2 MW | 165.7 MW at 3801 (merged at 0.81 pixel) |
| NOAA-21 Oct 22, site 957 | 3.0 MW, 2,712 K, M10+M11 | 2.2 MW, `pinned:two_band_range` |
| NOAA-21 Oct 30, sites 4279 / 4281 | 97.5 + 127.9 MW | 127.9 MW at 4281 only (N3) |
| NOAA-20 Oct 30, site 4267 | 19.7 MW | 37.0 MW, two sources 1.53 pixel apart across a scan boundary (copy or distinct, unresolved) |

### Expected 30-night effect versus v3 (inferred)

- Sol: pooled volume 10–20% lower; replacing 14 selected granules in the v3
  pool gives 11.20 → 9.89 BCM/yr (−11.6%), a non-representative sample.
- Opus: detected RH −17% to −22% (M7+M8 records −6.5 GW; RH75%/peak > 3
  records −3.4 to −3.9 GW; merges −6.2% and splits +1.9% over the dev
  granules); detections within ±1%; sum of site means about 770–830 MW
  (8.9–9.6 BCM/yr); matched total about 7.0–7.4 vs MYC 10.60 BCM/yr.

### Gates

- **Before the 30-night rerun:** N1, N2, N3, and a per-granule copy table
  (peak row/col/lat/lon, scan, footprint, RH75%, peak RH, `n_rh75`,
  temperature, bands, fit flag, cloud mask at the peak, component, M10
  prominence, merge group, assigned site). Cheap to add in the same change:
  N4 merge rule, N5 saddle test, N6 adjacency, N7 mean RH, D22 M7/M8 guard.
- **Before the backfill, the rerun must show:** 0 final M7/M8-only or
  M10+M7+M8 fits and 0 fits below 1,200 K; pinned fits ≤ about 4% (dev 3.6%,
  30/824); every cap exclusion and every RH75%/peak > 3 record listed and
  inspected; swath-edge share of same-pass detections at sites < 2 km apart
  falling from 66% (of 203 pairs) towards the all-detection 35%; swath-edge
  share of multi-source records near the all-detection share; cloud share of
  split sources reported; a gap in the copy-distance histogram; totals in the
  expected range; cross-satellite same-night pairs differing > 2× below v3's
  26–34%; site flips listed at MYC flares with ≥ 2 GFW sites within 2 km;
  complete satellite/granule coverage; no detections lost in dedupe.
- **Deferred:** optimal MYC matching (B11), association basins, cloud-window
  scaling, M11 and M12/M13 triggers, non-upstream flags, view-angle
  normalisation, empty-coverage logging.

## Main-session check of the v4 changes (2026-11-08)

GPT-6.1 Sol implemented the v4 fixes; the main session checked the diff and
the development outputs before the council review.

- The dev-night comparison used `outputs/dev_v3/` as the v3 baseline. That
  output predates the 10% model-error term: 103/369 eliminated fits against
  4/369 in final v3 on the same night. The tables in `decisions.md` and the
  README now use final v3 from the 30-night run.
- With rs2019's half-pixel spacing, swath-edge bow-tie copies 0.50–0.83 pixel
  apart stayed separate, and v4's per-site sum counted them twice: 22 sites
  had more than one source, 53% (25/47) of those records at the swath edge
  against 34% of all detections. The merge distance is now one pixel (D32):
  4 sites with more than one source, 12% (1/8) at the swath edge.

## Review 2: v3, 30 nights (2026-11-08)

Run reviewed: commit `414c198`, local nights 2026-10-06 – 11-04, 19,443
granules. Both reviewers ran the 36 unit tests (all pass) and opened raw
NOAA-20 granule `d20261010_t2330408`.

### Verdicts

- **Sol:** v3 improves v1 but remains a partial Nightfire implementation.
  Volumes and nightly "off" calls are not ready for operational use.
- **Opus:** v3 fixes 6 of the 9 review-1 findings in code; glow detections
  fell from 12.7% (501/3,935) to 0.97% (116/11,924). A new band-elimination
  failure produces single-pass fits of 300–3,127 MW. Regional totals and
  multi-week means at isolated persistent flares become usable once that fit
  path is closed; nightly radiant heat is not usable.

### Review-1 findings

| # | Finding | Status in v3 | Evidence |
| --- | --- | --- | --- |
| 1 | Site pixel on bow-tie fill | Fixed | `granule.py:88` requires finite M10. One NOAA-20 granule has 316,416 NaN M10 pixels and no fill geolocation. |
| 2 | Glow assigned to neighbours | Largely fixed | 0.97% (116/11,924) of detections within 3 km of a ≥5× brighter detection at another site, v1 12.7%. Bow-tie copies now split across sites (B5). |
| 3 | Saturation | Fixed | Raw radiance plus QF1 bits 2–3 (`granule.py:283–288`). |
| 4 | Window slice at index 1 | Fixed | `granule.py:120–121`. |
| 5 | Cloud rule | Implemented as worded in rs2021; neighbourhood size unvalidated | ≥6 clear neighbours clears 2.3% (89/3,929) of cloud-flagged detections; 74% (2,921/3,929) have 0 clear neighbours. Cloud-flagged detections hold 16% of detected radiant heat. |
| 6 | M10-only trigger | Not fixed | `granule.py:247–251`. |
| 7 | Fit quality | Partly fixed, one path worse | Weighting, bounds, convergence and `fit_flag` added; B1, B3, B4 below. |
| 8 | Validation matching | Fixed in structure | One-to-one at 2 km, pooled means; B11, B12 below. |
| 9 | Non-upstream flares | Not fixed | 136 flaring sites with no MYC match hold 2.83 of 11.18 BCM/yr (25%). Das Island (site 4267, 0.45 BCM/yr) has no match; Upper Zakum sites 4500 and 4452 (0.27, 0.37) pair with MYC sites of 8e-5 and 3e-5 BCM/yr. |
| – | Adjacent-granule duplicates | Fixed | 4,715 groups merged; B9 below. |

### New bugs

| ID | Bug | Location | Evidence | Found by |
| --- | --- | --- | --- | --- |
| B1 | Band elimination can drop M10–M13 and accept an M7+M8 fit | `nightfire.py:149–153` | 8 detections, all NOAA-20, 850–1,100 K, 6,755 MW together (8.8% of 76,629 MW detected), about 0.99 BCM/yr of the pooled total. Site 3802, 2026-10-10 23:30 UTC, peak (21, 3125): M7 = 0.122 vs M10 = 9.61 W m⁻² sr⁻¹ µm⁻¹; the 6-band fit lands at 1,271 K, elimination drops M13, M12, M11, M10; M7+M8 fit 916 K, 3,127 MW. Refit without M7: 1,699 K, reduced χ² 6.5, about 124 MW. *Checked:* 8 records. | Both |
| B2 | RH75% threshold set by the highest-RH fitted pixel, not the M10 peak | `granule.py:167` | Site 3804, 2026-11-03 22:41 UTC: peak pixel 71 MW (1,616 K, 184 m²), record 1,776 MW. 57 detections have RH75%/peak > 3: 4,197 MW vs 315 MW from their peak pixels. *Checked:* site 3804. | Opus |
| B3 | Two-band fits never face the χ² test | `nightfire.py:148` | χ²/max(n−2, 1) ≡ 0 for two bands. 2,107 two-band fits, 238 above 2,500 K. *Checked.* | Opus |
| B4 | Temperature floor 600 K; EOG catalogs keep T ≥ 1,200 K (rs2026) | `nightfire.py:101` | 76 detections below 1,200 K hold 9.0% of radiant heat. | Opus |
| B5 | Adjacent-scan copies of one flare can go to two sites | `granule.py:180–199` | One flare's scan-0 copy (105 MW) to site 3801, scan-1 copy to site 3802, 1.5 km apart. Same-pass co-detections at sites < 2 km apart are at the swath edge in 66% of 203 pairs vs 35% of all detections. | Opus |
| B6 | Connected flares merge into one cluster with one peak | `granule.py:112–143` | Synthetic row `[10,3,2,1,2,3,11]` yields one peak. Production frequency unmeasured. | Sol |
| B7 | Independent clusters at one site are discarded (max, not sum) | `granule.py:190` | 963 of 11,684 detections have more than one cluster; no spatial test separates scan copies from separate flares. | Sol |
| B8 | Exported row/col are the site pixel, not the flare peak | `granule.py:335–336` | Site 3802 anomaly exported at row 13, peak at row 21. | Sol |
| B9 | Deduplication can keep the non-detection | `pipeline.py:171–173` | 11 merged groups keep the record without the detection. | Both |
| B10 | `flaring_cloud` overrides `clear_no_flare`, README says `cloudy` | `pipeline.py:243` | 218 of 1,634 `flaring_cloud` nights have a clear pass and `rh_mw` = 0. | Opus |
| B11 | Greedy shortest-first matching mis-pairs large flares | `validate_myc25.py:96–100` | GFW site 989 (0.337 BCM/yr) takes MYC `s_5548` (0.009) at 484 m, leaving MYC `m_92` (0.367) paired with site 983 at 1,958 m. | Opus |
| B12 | Detection frequency compared under different definitions | `validate_myc25.py:58, 117` | See the detection-frequency table. | Both |

### Remaining deviations from the papers

| Step | Paper | Implementation | Measured or inferred effect |
| --- | --- | --- | --- |
| Trigger | rs2019, rs2023b: M10, M11, M12–M13 scattergram | M10 > zone mean + 4σ only | 59 of 356 persistent MYC sites have 0 detections; their MYC median is 0.002 BCM/yr (about 0.17 MW). |
| Zone statistics | rs2013 §2.1 | Five zones estimated separately; a 4σ clip replaces the DN > 100 screen | Bright sources can raise thresholds; size unmeasured. |
| MWIR background | rs2013: 10×10 window excluding hot pixels | Excludes M10-hot pixels only | M12 glow raises background mean and σ at large flares (inferred). |
| View-angle area | rs2021: area corrected to side view | None | Per-pixel RH relative to the site median: 1.21 at swath edge, 0.86 at nadir (221 sites, 7,104 detections). |
| Footprint | rs2013 Eqs. 4–7 | Neighbour pixel spacing | 0.573 km² at nadir, 2.60 km² at swath edge; EOG 0.742 × 0.776 km² at nadir. |
| Association | rs2026: flare-centred basins | Nearest GFW site within 2 km | At 45 persistent MYC flares 1–2 km from the matched site, per-pass rate 0.155 vs 0.72 in MYC. |
| Cloud clearing | rs2021: small isolated clouds over flares are cleared | 3×3 window, ≥6 clear neighbours | 37% of cloud-flagged detections have a mask-clear pass the same night vs 21% of cloud-flagged non-detections at the same sites (n = 3,813 / 20,201). |
| Fallback | rs2013 §2.5: 1,810 K for M10 + DNB only | 1,810 K after any rejected fit | 56–58 records. |

No atmospheric correction matches MYC25Q4, which uses VNF v3.0
top-of-atmosphere detections (rs2026).

### Results

- **Detections:** 11,924 after deduplication at solar zenith ≥ 97° (Opus);
  fit flags ok 11,688, eliminated 178, pinned 58. M10 in 99.9% of fits, M12 or
  M13 in 66.5%, two bands only in 17%.
- **Distributions (P5 / P50 / P95):** temperature 1,479 / 1,747 / 2,213 K;
  source area 0.26 / 2.18 / 53 m²; radiant heat 0.25 / 1.19 / 26 MW (P99 82 MW).
- **By satellite:** NOAA-20 4,546, NOAA-21 4,146, SNPP 3,232 (no SNPP data
  after 11-02). Detections per clear pass 2.10%, 2.02%, 1.78%.
- **Cross-satellite, same site and night, both passes clear:** when one
  satellite detects, the other detects in 68–78%; RH ratio median 0.94–1.05,
  log-log r 0.83–0.88; 26–34% of pairs differ by more than 2× (n = 1,294–1,604
  per satellite pair).
- **Possible false positives:** 88 sites have exactly one detection in 30
  nights (59% single-pixel, 32 with a MYC match). 167 of 506 sites with clear
  detections get more than half their 30-night RH from one pass. 10 of the 88
  single-detection sites fall in the South Atlantic Anomaly box, no excess.
- **MYC25Q4 2025, 712 matched sites:** 8.35 vs 10.60 BCM/yr; 7.27 (Opus,
  removing 135 suspect detections) to 7.48 (Sol, removing the 8 M7+M8 fits).
  After Opus's removal, by regime: persistent 5.13 vs 9.89 (n = 356),
  intermittent 1.78 vs 0.66 (n = 171), sporadic 0.31 vs 0.06 (n = 67); by
  region: Persian Gulf 3.01 vs 3.81 (3.70 before removal), Gulf of Mexico 1.16
  vs 1.79, Gulf of Guinea 1.33 vs 2.71. Positive-positive log-log r 0.51
  (n = 365), median ratio 0.66.
- **Largest sites:** site 3802 falls from 0.75 to 0.32 BCM/yr without its
  3,127 MW pass, site 876 from 0.28 to 0.03. Sites 4279 (0.67 vs 0.73) and
  3205 (0.19 vs 0.19) agree with MYC. MYC's largest matched flare, `s_9435`
  (1.36), gets 0.30 at site 3779, 1.2 km away.

### Detection frequency at persistent MYC sites (n = 356)

| Definition | Median detections / clear passes |
| --- | --- |
| v3 validation: own site, strict clear rule | 0.269 |
| Any detection makes its pass clear | 0.388 |
| Detection at any GFW site within 2 km of the MYC flare, strict clear | 0.380 |
| Both of the above | 0.509 |
| Pooled sites, all detections / strict-clear passes | 0.614 |
| MYC `n_det / n_clear` | 0.628 |

In 355,557 MYC site-years `n_det` never exceeds `n_clear`, so MYC counts only
detections on passes it calls clear; 0.51 is the closest like-for-like
comparison. This corrects the review-1 note that MYC counts detections whether
or not the pass is clear. Coverage and raw counts agree within about 10%: 1.47
vs 1.59 clear passes per night, 0.50 vs 0.56 detections per night. Opus splits
the 0.36 gap into roughly equal parts: cloud clearing, site association, and
small flares plus real 2025→2026 change. Detection per clear pass is 0.39 near
nadir and 0.28–0.29 at the swath edge (17,726 clear passes); MYC samples the
same geometry.

### Usability

| Use | Sol | Opus |
| --- | --- | --- |
| Regional totals | With caveats, for screening | With caveats, after B1–B2 and non-upstream flags |
| Per-site multi-week or annual volume | Not usable quantitatively | With caveats, isolated persistent flares only |
| Per-site nightly radiant heat | With caveats, isolated confirmed detections | Not usable: about 2.8 clear passes per night, > 2× same-night spread in 26–34% of pairs |
| Nightly on/off | "On" usable, "off" not | "On" with caveats, "off" not: 59% of flaring nights with ≥ 2 clear passes include a clear non-detection (n = 3,436) |

### Fix order agreed after review 2

| Order | Fix | Addresses | Planned for |
| --- | --- | --- | --- |
| 1 | Fit guardrails: M10 or M11 in the final band set, low-temperature handling, better elimination, bounded two-band fits | B1, B3, B4 | v4 (D22) |
| 2 | RH75% threshold from the M10 peak; record peak RH and pixels summed | B2 | v4 (D23) |
| 3 | Merge adjacent-scan copies before site assignment; split clusters with separate maxima; keep independent flares at one site | B5, B6, B7 | v4 (D24, D25, D32) |
| 4 | Flare-centred association basins; optimal matching | Association, B11 | later |
| 5 | Footprint-scaled cloud clearing, calibrated against other satellites' same-night passes | Cloud deviation | later |
| 6 | M11 and M12/M13 triggers | Finding 6 | later |
| 7 | Non-upstream flags, upstream-only totals | Finding 9 | later |
| 8 | Export peak location; prefer detected record in dedupe; settle `flaring_cloud` precedence; detection frequency under MYC's definition | B8, B9, B10, B12 | v4 (D26–D29) |
| – | View-angle area normalisation | Deviation | later |

## Review 1: v1, 7 nights (2026-11-07)

GPT-6.1 Sol and Claude Opus 5.5 reviewed the 7-night v1 run independently
against the papers. Both judged regional totals and persistent flares usable,
and per-site nightly volumes and the flaring status of sites with another site
within 2 km (3,858 of 7,637) not yet reliable. Opus 5.5 reproduced site 4279's
radiant heat (93.5 MW) from raw NOAA radiances.

Fixes, with their status in v3 (commit `4238bf1`, implemented in three rounds
with GPT-6.1 Sol):

1. Applied. Site pixels need finite M10 radiance (v1: 11% of site-overpasses,
   21,205 / 189,764, used bow-tie-deleted pixels); passes repeated in adjacent
   granules (<10 s) are deduplicated.
2. Applied. Scan-separated 8-connected clusters, kept only when the peak is the
   brightest M10 pixel within 2 pixels; RH75% per cluster; each cluster to one
   site; per-site value is the strongest cluster (v1: 12.7% of detections,
   501/3,935, sat within 3 km of a ≥5× brighter detection at another site).
3. Applied. Saturation from raw radiance and QF1 bits 2-3; a saturated M10 no
   longer drops the pixel.
4. Applied. ±2-pixel window clipped at the array edge.
5. Applied. A cloud-flagged detection counts as clear only with ≥6 of 8 clear
   neighbours; otherwise the night is `flaring_cloud` (lit, no clear-sky value).
   `*_detclear` columns keep the v1 rule.
6. Open. M11-only and M12+M13 triggers.
7. Applied. Noise-weighted fit with a 10% relative model error (without it,
   100% of fits above 20 MW eliminated bands on model error), residual band
   elimination, physical/convergence rejection, `fit_flag`; M7/M8 in stage 2.
8. Applied. One-to-one matching at 2 km, pooled clear-sky means, per-pass
   detection frequency, regional totals (`validation_summary_<tag>.json`).
9. Open. Flag non-upstream flares (for example Das Island, Khafji).

v1 → v3 over the same 30 nights:

| Measure | v1 | v3 |
| --- | --- | --- |
| Flaring site-nights | 7,691 | 4,060 clear + 1,634 lit under cloud |
| Sites flaring on ≥1 night | 758 | 504 clear, 558 incl. cloud |
| Sum of 30-night site means | 915 MW (10.6 BCM/yr) | 944 MW (11.0 BCM/yr) |
| Detection rows | 16,323 | 11,967 (33% cloud-flagged) |
| Fit flags | none recorded | ok 98% (11,731), eliminated 1.5% (178), pinned 0.5% (58) |
| Fits with M12 or M13 | 63% | 67% (M7/M8 in 66%) |
| Temperature P5 / P50 / P95 | 1452 / 1797 / 2375 K | 1479 / 1748 / 2213 K |
| Matched sites, 2 km one-to-one: ours vs MYC 2025 | n/a | 8.35 vs 10.60 BCM/yr (n = 712) |

Matched regional totals (v3, BCM/yr, ours vs MYC 2025): Persian Gulf 3.70 vs
3.81, Gulf of Guinea 1.43 vs 2.71, Gulf of Mexico 1.45 vs 1.79, South China
Sea 0.31 vs 0.57, Malaysia & Indonesia 0.45 vs 0.56.

Largest open gap: at the 356 matched persistent flares the median share of
clear passes with a detection is 0.27 here vs 0.63 in MYC25Q4 (`n_det /
n_clear`). Candidate causes: MYC counts detections whether or not the pass is
clear, VNF triggers on M11 and M12+M13 as well as M10 (fix 6), and MYC
association basins are wider than the 2 km nearest-site rule.
