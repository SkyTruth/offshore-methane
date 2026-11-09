# Sol Round 1 analyses

Run from `nightly-flaring/`, with no installs or external requests:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/prepare.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/explore.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/spatial_and_peers.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/validate_candidates.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/final_checks.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/aperture_sensitivity.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/temporal_and_local.py
```

All writes go to this directory. `prepare.py` imports `build_viewer` and calls
`load_segments(find_segments())`; it never calls the viewer builder or catalogue
loader. Duplicates and strict-clear diagnostics use the pipeline functions.
The probability-screen denominator uses raw cloud flags 0/1, without the
outcome-dependent isolated-cloud detection override. The local date is
`(time_utc + lon/15 hours - 12 hours).date()`. The baseline is August 10 through
September 7, with September 8 through November 4 used for the later run screen.

The preparation scripts generate six intermediate `.pkl` files totalling about
564 MB. These are disposable caches and are removed from the delivered research
folder; rerunning the sequence regenerates them. CSV/JSON summaries are retained.
No original data or project code is edited.

Primary outputs:

- `preparation.json`: data coverage and graph-component counts.
- `eligibility.csv`, `anchors.csv`: baseline screen and selected site cohort.
- `runs.csv`: all site-level detection-free runs after the baseline; `bonf` is
  `min(1, tail * 21 * 58)` under the constant independent-night model.
- `spatial_runs.csv`, `source_radius_runs.csv`: source-position checks and
  aperture-level recounts. This is a proxy, not a fitted source-mixture model.
- `candidate_validation.json`: 20,000 posterior-predictive null simulations,
  seed 919067. Each simulation draws a probability per anchor and then simulates
  independent qualified nights. Nonqualified nights never reset the simulated
  run, giving an upper envelope relative to observed nonqualified positive
  resets. These are model-based false alerts, not measured methane false alarms.
- `final_candidate_gates.csv`, `local_donors.csv`: preceding-28-night baselines,
  recent heat and donor activity, collapsing donor components per night.
- `injection_power.csv`, `required_qnights.csv`: ideal all-evidence erasures in
  every possible 4-, 7-, and 14-calendar-night evaluation window. Only the window's
  qualified opportunities can accumulate evidence. This uses the actual pass and
  cloud schedule; it does not test source tracking, causal classification, or the
  observation model under a reduced physical flare.
- `aperture_sensitivity.csv`: 0.5–3-km radii and pre-gap source heat. Positive
  evidence includes cloudy groups; heat summaries use copy-peak cloud flags 0/1.
- `4582_nearby_groups.csv`: all groups within 3 km during the fifteen-night
  site-level gap, including the weak November 4 source assigned to 4583.
- `heat_loss_onsets.csv`: source heat summaries around the inspected August 29,
  September 13 and October 21 changes; these dates were inspected retrospectively,
  not estimated by a fitted change-point algorithm.

The data contain no production/venting labels. Neither internal simulations nor
held-out passes can estimate the probability that a flagged site vented methane.

## Round 2 cross-checks

The Round 2 scripts read the already supplied pass/statistics files in
`scripts_opus/cache/`, the original two copy CSVs, and the local GFW site table.
They never modify those caches and never read
`myc_match.pkl`. Do not rerun Opus's cache-builder to reproduce Round 2: its
annual-catalogue step is unnecessary. The cached Opus pass table has 2,417,675
rows; Sol's Round 1 SZA-before-deduplication convention retained five more rows.
All three Opus run-count thresholds reproduce exactly from the Opus cache.

The raw-file audit found 228 source groups missing from the Opus copy cache:
39,314 original versus 39,086 cached groups before local-night filtering,
39,149 versus 38,921 during the 88-night window. Its component/time deduplication
can drop split maxima belonging to different group IDs. All final Round 2
spatial results therefore read the original two copy CSVs. Among the common
groups, location, radiant heat and assignment match; 22 retained representative
times differ by one second. The 3,548 selected pass records at 11 sites match
the original overpass CSVs in every compared field.

Run from `nightly-flaring/`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_calibration.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_stationary.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_evidence.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_dependence.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_regional.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_entities.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_q1_screen.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_primary_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r2_partner_spatial.py
```

Primary records:

- `r2_calibration.json`: exact run-score replication, 20 within-zone pass
  shuffles, and 100 each of one-, three-, and seven-calendar-night block
  shuffles. Blocks carry whole outcome/geometry sequences, including C/U.
  They retain actual operating changes; these are dependence stress tests,
  not labelled benign nulls or empirical false-alarm rates.
- `r2_stationary.json`: 100 finer-stratum pass shuffles within site × zone ×
  satellite × neighbour-clear category, keeping C/U positions fixed.
- `r2_evidence.json`, `r2_candidates.json`, `r2_candidate_nearby_groups.csv`,
  `r2_candidate_weekly.csv`: candidate identities, source-centred searches,
  recent heat, quiet-site selection diagnostics, and qualification counts.
  Candidate pass-gap starts may fall after a positive observation on the same
  local night; 927's first fully dark night is October 30, not October 29.
  Separate pre-week summaries are retained for the pass-gap date and the first
  fully dark site night. The memo's candidate table uses the latter (927:
  October 23–29, median 9.44 MW from 11 clear source groups).
- `r2_bright_misses.csv`: all 94 nadir clear misses at the 12 retrospectively
  selected >10-MW steady sites. Searches match satellite and ±10 seconds;
  source-centred apertures use only Aug 10–Sep 7 positions. Nearby groups are
  evidence of attribution ambiguity, not proof of identical physical flares.
  `n_hot_m10` is conditioned on confirmed assigned sources in `granule.py`;
  its zero value at a nondetection contains no rejected-trigger information.
- `r2_dependence.json`, `r2_markov_candidates.csv`: 20,000 within-site
  baseline permutations and 20,000 simulations for each stationary
  qualified-opportunity Markov stress model. Each site's miss fraction is
  drawn from its frozen beta posterior; the miss persistence is varied.
  Nonqualified positive resets are omitted, as in the original conservative
  envelope. Pooled persistence cannot be identified as a benign-state rate.
  `r2_markov_sensitivity.csv` tabulates the same tail formula at r=0.15,
  7/19, the Beta(8,13) upper 97.5th percentile, and 0.60.
  `r2_aperture_baseline_dependence.csv` repeats baseline adjacency counts on
  fixed 0.75/1/1.6-km source apertures at the same 21 anchors. This is a
  localization sensitivity, not a fitted or independently selected source
  cohort; hard aperture misses can create serial structure themselves.
- `r2_regional.json`, `r2_gom_*_weekly.csv`, `r2_gom_windows.csv`: descriptive
  Gulf of Mexico comparisons. The logistic fit uses all 88 nights and is
  retrospective, with site indicators, zones, satellite and neighbour cloud
  flags. There are no weather labels. Partial first/last weeks remain marked
  by their calendar week; memo comparisons use complete weeks only.
- `r2_entities.json`, `r2_entities.csv`, `r2_source_eligibility.csv`,
  `r2_source_runs.csv`: baseline-only DBSCAN feasibility. The diameter ≤2 km
  and neighbour distance ≥2 km guard is an exploratory conservative proxy,
  not a fitted positional-error model. A separate r95 ≤1 km sensitivity is
  recorded; it is not an independently selected threshold. Negatives use
  the nearest site's cloud pixel, not a measured source-centre cloud pixel.
  `r2_source_guard_sensitivity.csv` records the full exploratory grid of
  diameter and r95 guards, with non-overlapping-aperture separation guards.
- `r2_source_pairs.csv`, `r2_4582_4583_multigroup_passes.csv`: assignment
  overlap and simultaneous-source checks. The 4582/4583 neighbourhood has
  97 groups in 94 baseline same-satellite passes. Three have two resolved
  groups; none has two groups ≥3 MW. This establishes ambiguity in treating
  their site records as independent bright-source histories; it does not
  prove there is only one physical flame.
- `r2_primary_check.json`, `r2_copy_groups_missing_from_opus_cache.csv`:
  direct raw-file checks, the missing-group inventory, and compared fields.
- `r2_partner_spatial.json`, `r2_partner_spatial.csv`: all 84 Opus flags checked
  against the original grouped copies. The flags with source evidence within
  2 km (11/84), and separately at 2–4 km (25/84), match the original memo despite
  the incomplete copy cache. Counts describe nearby evidence, not a same-source
  match or a blanket veto.
- `r2_eligibility_q1.csv`, `r2_eligibility_q2.csv`, `r2_q1_runs.csv`: one-clear-
  pass eligibility sensitivity. The full 24-site one-pass family gives 927
  a nominal adjusted tail of 0.0513047 after six qualified misses; do not
  substitute a post-hoc smaller source family to declare a crossing.

The new scripts write only small CSV/JSON summaries under `scripts_sol/`.
All episode scores, thresholds, state definitions and feasibility comparisons
are retrospective. No new untouched confirmatory sample was acquired.

## Round 3 reconciliation and audits

Run from `nightly-flaring/` in this order (no installs or downloads):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r3_checks.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r3_calibration_audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r3_local_heat.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/flareout_research/scripts_sol/r3_raw_audit.py
```

- `r3_checks.json`, `r3_transitions.csv`, `r3_source_transitions.csv`: Q-only,
  Q-or-positive, and Q-trial/non-Q-positive-reset transition conventions.
  The source proxy uses 1-km apertures at the frozen site anchors; it is not
  a reselected, validated source cohort. The 20,000 baseline-resampling draws
  retain whole calendar-night fleet vectors and use circular blocks of 1/3/7
  nights. The original independent score/eligibility is frozen. These are
  stress tests; sparse baseline support and unlabelled operating changes
  preclude treating them as empirical benign false-alert rates.
  `r3_duration_diagnostic.csv` counts available transitions after miss runs of
  one, two, three, or >=4 qualified opportunities; only two anchor transitions
  follow a third miss, with no observations of longer preceding runs.
- `r3_markov_tails.csv`, `r3_calibration_audit.csv`: posterior-predictive tail
  sensitivity with independent Beta(1,1) transition priors, pooled versus
  site-specific entry rates, alternative persistence pools, and boundary versus
  after-positive starts. Values above 1 are left untruncated to show the
  expected-count/union-bound scale; none is a calibrated venting probability.
  The 1-km-source row remains a diagnostic, with localization failures included.
- `r3_observed_eligibility.csv`, `.json`, `r3_observed_cohort_transitions.csv`:
  Opus's literal observed-night rule, using >=12 observed baseline nights,
  lit fraction >=0.8, and >=3-MW median raw-clear positive-pass heat. This gives
  25 eligible site records, requiring a different family from the 21-site rule.
- `r3_source_daily.csv`, `r3_source_windows.csv`, `r3_radius_gaps.csv`,
  `r3_late_source_groups.csv`: original-copy evidence for 4582/4583,
  4497/4485 and 4500; baseline-only centres; radius sensitivity. Overlapping
  apertures cannot establish independent flames even if one group is bright
  in both apertures. The 4582/4497 apertures are disjoint and have simultaneously
  bright sources on 84 of 105 baseline passes.
- `r3_off_low_sensitivity.csv`: assumed independent-pass likelihood ratios for
  off versus continuing weak-flame detectability. These are model assumptions,
  not labelled off/low validation data; dependence would require a new likelihood.
- `r3_local_heat.json`, `r3_4500_heat_controls.csv`, `r3_4500_groups.csv`,
  `r3_4500_weekly.csv`: newly checked nearby thermal co-change. The October 22
  reduction persists in each satellite, nadir/high-clear-neighbour groups,
  and 0.5/0.75/1/1.5-km apertures, despite daily detections. This does not
  establish shared infrastructure, continued production, or a common cause.
- `r3_raw_audit.json`: direct comparison of the newly inspected 4485/4500/
  4515/4561/4575 pass histories with original overpass CSVs, plus the >=97-degree
  SZA check at source centroids contributing to the 4500 October/November audit.
  One unassigned source has no nominal 4500 pixel in its granule; source
  SZA is calculated from the supplied local `nightfire.solar_zenith_deg`,
  preserving positive evidence even without that platform's exposure row.

`sol_r3.md` contains the shortlist decisions and final recommendations from Sol's
side. Round 3 adds no held-out confirmatory observations or causal labels.
