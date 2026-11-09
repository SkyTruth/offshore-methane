# Opus round 1: modelling flares that go out

Scripts: `scripts_opus/s00`–`s08` (run from `nightly-flaring/`). Nights
2026-08-09 to 11-04.

## The result that sets the design

A per-site test that treats passes as independent draws at the site's past
detection rate flags 84 dark runs at 51 sites at P < 1e-4, about one a night.
Shuffling each site's clear outcomes within view zone gives 1.6 such runs
(mean of 20 shuffles, max 4). On/off spells and turndowns are the norm:
50 of the 84 runs sit at sites with 2 or more flagged runs, and 35 of 74
closed runs end with the flare back at under half its earlier radiant heat.
The model has to learn what a normal off spell looks like at each site.

Definitions. Passes with SZA ≥ 97°, `pipeline.dedupe_overpasses`, clear by
`pipeline.strict_clear`. Each pass is D (clear detection), Z (clear, no
detection), C (detection under cloud: lit) or U (cloudy, nothing seen). View
zone from `col`: nadir 1008–2191, mid 640–1007 and 2192–2559, edge
otherwise. Monitored sites: ≥ 6 clear detections, clear-pass detection
fraction ≥ 0.3 (221 sites). Steady sites: ≥ 40 clear passes, fraction ≥ 0.5
(126 sites).

Per-pass detection at steady sites (18,517 clear passes; selected on
fraction ≥ 0.5, so above the brief's 0.50 like-for-like figure):

| Site median RH | nadir | mid | edge | sites |
|---|---|---|---|---|
| < 1 MW | 0.81 | 0.75 | 0.64 | 45 |
| 1–3 MW | 0.85 | 0.79 | 0.70 | 52 |
| 3–10 MW | 0.88 | 0.84 | 0.77 | 17 |
| > 10 MW | 0.89 | 0.83 | 0.81 | 12 |

Logit effects (site fixed effect): mid −0.47, edge −1.03 (vs nadir);
NOAA-21 −0.11, SNPP −0.43 (vs NOAA-20); < 8 clear neighbours −0.28. Flares
above 10 MW are missed on 11% of nadir passes.

## Idea 1. Dark-run score (Bernoulli CUSUM) against the site's own past

1. **Detects** a run of clear passes with no detection at a site that was
   detectable before. H0: still lit at the baseline per-pass probability,
   passes independent given geometry.
2. **Model.** `logit p_ij = logit f_i + β_zone(j) + β_sat(j) + β_cn(j) − c_i`;
   `f_i = D/(D+Z)` over clear passes before the run, `c_i` centres the
   offsets on the baseline's geometry mix. CUSUM (Page 1954) with off-state
   detection rate ε → 0: `S_t = max(0, S_{t−1} + ℓ_t)`, `ℓ = −log(1 − p_ij)`
   for Z, reset on D or C, 0 for U. Alarm at `S ≥ ln(1/α)`; under H0
   `P(run) = Π(1 − p_ij) = e^{−S}`.
3. **Columns.** `overpasses`: satellite, col, sza, cloud_mask,
   cloud_clear_neighbors, detected, time_utc, site_id (`s02_runs.py`).
4. **Confounders.** Geometry and satellite enter `p_ij`; the SNPP outage
   only removes passes. Clear misses inside flagged runs are not enriched in
   "probably clear" masks (4.6% vs 6.4% outside; n = 2,124 vs 4,129) or edge
   passes (42% vs 48%). 75 of 84 runs have ≥ 3 clear misses from ≥ 2
   satellites. Benign stops and turndowns are left to ideas 2–4.
5. **Feasibility.**

   | α | observed runs (sites) | shuffle null runs, mean (max) |
   |---|---|---|
   | 1e-3 | 130 (70) | 12.1 (18) |
   | 1e-4 | 84 (51) | 1.6 (4) |
   | 1e-5 | 60 (39) | 0.3 (2) |

   Synthetic flare-outs (3 random starts per steady site; later Z stay Z,
   D become Z, C become U), time to alarm at α = 1e-4:

   | Size | alarm by 11-04 | median nights | 90th pct |
   |---|---|---|---|
   | < 1 MW (n=120) | 92% | 4 | 15 |
   | 1–3 MW (n=126) | 99% | 5 | 19 |
   | > 3 MW (n=78) | 99% | 3 | 10 |

6. **Reliability.** Power from injection. The shuffle null gives false
   alarms under independence, a lower bound on false alarms under benign
   stops.
7. **Cannot** separate a flare-out from a benign stop or from a turndown
   below the floor.

## Idea 2. Lit / low / off hidden semi-Markov model per site

1. **Detects** an off spell longer than the site's own and its class's off
   spells. H0: the current dark spell is a draw from the site's sojourn
   distribution in "off" or "low".
2. **Model.** States L (lit), W (low), O (off). Clear pass:
   `P(D | k, j) = expit(a_ik + β·x_j)`, `a_iW` constrained to p ≤ 0.3,
   `P(D | O) = ε` (from detections at sites with ≤ 1 detection in 88
   nights). Cloudy pass: `P(C | L) = c_i`, `P(C | O) = ε_c`. Intensity for D:
   `log RH_ij ~ N(μ_ik + γ·edge_j, σ²)`, left-censored at a zone floor.
   Continuous-time hazards `λ_{i,k→k'}` between passes,
   `log λ_i ~ N(μ_region, τ²)`; Weibull off sojourns (Yu 2010). Alert:
   `P(O for ≥ D nights | data) > 0.95` and Bayes factor O:W > 100.
3. **Columns.** As idea 1, plus `rh_mw`.
4. **Confounders.** Turndown first. Site 4497 fell from median 26.7 MW to
   1.09 MW around 8 September; per-pass detection went from 0.82 / 0.79 /
   0.64 (nadir / mid / edge; n = 28 / 28 / 36) to 0.18 / 0.21 / 0.09
   (n = 62 / 56 / 85), and idea 1 flags 7 runs there. After n clear misses
   the O:W Bayes factor is `((1 − ε)/(1 − p_W))^n`; with p_W = 0.15 and
   ε = 0.002, 100:1 needs 29 clear passes, about 15 clear nights.
5. **Feasibility.** Of 221 monitored sites, dark spells with ≥ 10 clear
   passes reach ≥ 7 nights at 31, ≥ 14 at 18, ≥ 28 at 2. Spells ongoing at
   11-04 that the model would rank first:

   | Site | Region | Record | Clear passes in spell |
   |---|---|---|---|
   | 3662 | Persian Gulf | flaring 34 nights (median 0.75 MW), dark 09-19 to 11-04 | 148 |
   | 3644 | Persian Gulf | flaring 33 nights (0.46 MW), dark 09-16 to 11-04 | 159 |
   | 6011 | Bohai | flaring 16 nights (0.55 MW), dark 09-16 to 11-04 | 126 |
   | 927 | Gulf of Mexico | 6.3 MW, ≥ 98% of clear passes detected, dark from 10-29 | 15 |

   3662 and 3644 (46 km apart): no site within 2 km, no hot source within
   4 km during the spell; both under 1 MW, so turndown below the floor fits
   as well as an outage. 927: a large, near-continuous flare, 7 nights
   dark.
6. **Reliability.** Log score on 20% held-out passes vs two-state and
   stationary models; O-vs-W confusion on injected outages and turndowns
   (real detections thinned by the size-dependent p).
7. **Cannot** tell "off" from a flame below about 0.3 MW. A site's own
   sojourn distribution rests on 0–7 spells in 88 nights; the regional prior
   carries most of the weight.

## Idea 3. Flare-centred units instead of GFW sites

1. **Detects** outages of a flare location, so a flare reassigned to a
   neighbour, or moved beyond 2 km, is not an outage. H0: the flare entity
   is still lit, wherever its detections are assigned.
2. **Model.** DBSCAN (haversine, 750 m) on detection positions
   (`peak_lat/peak_lon` or copy `centroid_lat/lon`). A pass is D for an
   entity if a detection falls within `r_j` of its centre (0.75 km nadir to
   1.6 km edge), Z if a member site is clear with none. Same-satellite
   passes within 10 s count once (D28). Run ideas 1–2 on entities.
3. **Columns.** `overpasses.peak_lat/peak_lon`; `copies.centroid_lat/lon`,
   `assigned_site`, `site_dist_m` (`s05_checks.py`, `s07_entities.py`).
4. **Confounders.** 33,699 of 34,087 detections fall in 405 entities; 94
   entities draw detections assigned to more than one site, 51 with no site
   holding 90%. DBSCAN chains through dense fields (112 sites in a 22 km
   square off Abu Dhabi, explainer Figure 12), so entity diameter needs a
   cap.
5. **Feasibility.** Among the 84 flagged runs, a neighbouring site was lit
   during the run in 3, a hot source lay within 2 km in 11 and at 2–4 km in
   25. Site 6969 (Gulf of Guinea, dark from 10-21) had a neighbour lit 15
   times and 19 sources within 2 km during its run: a reassignment.
6. **Reliability.** Site-level minus entity-level alerts should all be
   swaps visible in `copies`.
7. **Cannot** separate two flares on one platform or recover a flare never
   detected.

## Idea 4. Common-mode terms for regional and field-level stops

1. **Detects** stops shared by nearby sites (shut-in, storm, field
   maintenance, field-wide failure). H0: site outcomes are independent
   given site and geometry.
2. **Model.** `logit p_ijt = a_i + β·x_j + u_{r,t} + v_{g,t}`; `u` a
   regional random walk over nights, `v` a term for sites within 10 km.
   Site alerts are scored on the residual; a co-stop moves an alert to
   "likely shut-in".
3. **Columns.** As idea 1, plus `regions.region_of(lat, lon)`.
4. **Confounders.** Regional haze or cloud-mask misses are absorbed with
   the shut-ins.
5. **Feasibility.** Nightly regional residuals are overdispersed: sd 0.051
   vs 0.041 binomial in the Persian Gulf (median 40 monitored sites a
   night), 0.108 vs 0.079 in the Gulf of Mexico (16). Region-nights with
   z < −3: 19 of 460 (0.6 expected); z > 3: 15. Persian Gulf weekly
   detection stays 0.61–0.69: no storm-scale stop in 88 nights. Co-stops:
   9 of 84 runs had another flagged run within 10 km starting within 3
   nights, vs 4.1 expected (95th percentile 8) with starts jittered ±40
   nights. Example: 4582 (10.9 MW, dark from 10-20), 4583 (2.4 km, from
   10-21), 4497 (from 10-22).
6. **Reliability.** Inject a regional stop (all monitored sites in a 50 km
   box off for 5 nights); `u` should absorb it with no site alert left.
7. **Cannot** distinguish a shut-in from simultaneous failures.

## Triage for any alert

1. Automatic: entity check (3), ≥ 2 satellites with clear misses,
   common-mode check (4), O:W Bayes factor (2).
2. Analyst: M10 chips for the spell's clear passes and the last lit passes;
   pre-spell radiant heat (4497-style fade vs 927-style stop).
3. Outside checks last: methane plume search (TROPOMI, EMIT, Sentinel-2),
   operator notices (Plant et al. 2022; Irakulis-Loitxate et al. 2022).

## What none of these can tell us

- Off vs a flame below the floor (detections thin out below about 0.3 MW).
- Venting vs shut-in: the data measure heat, not methane or production.
- Daytime outages: all passes fall 00:50–02:50 local.
- Outages shorter than about 3–6 nights; sites never detected (no baseline).

## Questions for my partner

1. Do you reproduce 84 runs vs 1.6 under the shuffle null? Does the
   within-zone shuffle understate the null (autocorrelated cloud, geometry)?
2. What ε (detections at an unlit site) do you estimate, and from where?
3. Check 3662, 3644, 6011 and 927, including hot sources beyond 2 km.
4. Steady flares above 10 MW are missed on 11% of nadir clear passes: cloud
   mask misses, flame variability, or the pipeline? Can `n_hot_m10` or the
   copy table tell?
5. Do you see a Gulf of Mexico storm signature in any subset of sites?
6. What alert budget should the threshold target? At α = 1e-4, idea 1
   alone gives about 29 alerts a month across 221 monitored sites.

## References

Page 1954, Biometrika 41:100 (CUSUM). Yu 2010, Artificial Intelligence
174:215 (hidden semi-Markov models). Plant et al. 2022, Science 377:1566
(unlit and inefficient flares). Irakulis-Loitxate et al. 2022, Environ. Sci.
Technol. Lett. 9:520 (offshore Gulf of Mexico emission event).
