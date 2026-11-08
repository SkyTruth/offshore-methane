# Reviews

Each review is a two-model council: GPT-6.1 Sol and Claude Opus 5.5 get the
same brief (code, papers in the scratch text extractions, run outputs) and
report independently. Counts below are the reviewers' measurements unless
marked *checked*, which means the main session re-derived them from the
outputs. Choices made in response are recorded in [decisions.md](decisions.md).

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
