# Brief: modelling flares that go out, from the nightly flaring dataset

You are one of two independent researchers (GPT-6.1 Sol and Claude Opus 5.5).
You will work in rounds and exchange written memos through this folder.

## The question

SkyTruth cares most about **flares that go out**: offshore sites that should be
flaring but are not, because an unlit flare (or a flare-system failure) can
mean gas is vented, possibly as a methane super-emission event.

**How can we use this nightly VIIRS flaring dataset to build a model that
identifies infrastructure that should be flaring but isn't?** Focus on what
this dataset can support: per-site, per-pass, per-night detection histories
at 7,637 offshore oil sites over 88 nights. The model must come from this
nightly flaring data alone. Outside sources may appear only as a later
validation step, and briefly.

Be creative, but rigorous and conservative. The cost of a false alarm is high
(SkyTruth's credibility with a public accusation); the model must separate a
real flare-out from the many benign or artefactual ways a flare can drop out
of the record:
- cloud, partial cloud, cloud-mask errors; no pass that night (gaps, SNPP outage);
- detection limit: a flare that shrank below the ~1 MW detection floor vs one that
  went out; view-angle dilution at the swath edge (edge/nadir radiant heat 1.58,
  n=39); per-pass detection probability at persistent flares ~0.50 here
  (like-for-like clears) vs 0.63 in EOG's catalogue, so single missed passes are
  routine;
- the flare assigned to a neighbouring site (3,858 of 7,637 sites have another
  site within 2 km; 64% (126/197) of neighbouring flaring pairs swap the flare
  between them at least once);
- benign stops: production shut-ins (Gulf of Mexico hurricane season is Aug-Oct,
  and storms shut in many platforms at once), maintenance, flare-gas recovery
  projects, decommissioning, gas re-injection;
- intermittent flaring by design (EOG classes sites persistent / intermittent /
  sporadic).

## What we want from you

Ideas for **models and detection rules** that use this dataset, each with:
1. what it detects and the null hypothesis it tests;
2. the statistical model (state the equations you would use; e.g. per-site
   detection probability given clear-sky, view angle, satellite and flare size;
   change-point or run-length tests; hierarchical/regional terms for correlated
   shut-ins);
3. how to compute it from the files below (columns named);
4. confounders and how each is controlled, with numbers from the data where you
   can get them;
5. a feasibility check on the 88 nights: e.g. how many sites/events it flags at
   a stated threshold, the expected false-alarm count under the null, and a few
   example site IDs with what their record shows (you may run quick analyses);
6. how a flagged event would be triaged, and how the method's reliability can
   be measured from the nightly data itself (e.g. held-out passes, synthetic
   flare-outs injected into real site histories, stability across satellites);
7. what it cannot tell us.

## Data (all local; do not download satellite data)

Worktree: `/Users/jonathanraphael/.t3/worktrees/offshore-methane/t3-d5acd8c5`.
Project folder: `nightly-flaring/`. Python: `nightly-flaring/.venv/bin/python`.

Two run segments cover local nights 2026-08-09 to 2026-11-04 (88 nights):
- `nightly-flaring/outputs/v5_backfill/{site_nights,overpasses,copies}_2026-08-09_2026-10-05.csv`
- `nightly-flaring/outputs/v5_30n/{site_nights,overpasses,copies}_2026-10-06_2026-11-04.csv`
- Load both with `import build_viewer as b; nights, over = b.load_segments(b.find_segments())`
  (run from `nightly-flaring/`; takes ~1 min; overpasses ~1M rows per segment).
- Sites: `nightly-flaring/outputs/gfw_oil_sites.csv` (site_id, lat, lon,
  n_structures, structure_ids; GFW fixed infrastructure labelled oil, merged
  within 750 m).
- EOG MYC25Q4 annual catalogue match: `validate_myc25.match_sites(nights,
  validate_myc25.load_myc(2025), 2000.0, over)` gives each site's matched EOG
  flare, its 2025 regime (persistent/intermittent/sporadic/undetected) and BCM.

`site_nights` (one row per site per local night): night, site_id, lat, lon,
status (flaring | flaring_cloud | clear_no_flare | cloudy | not_observed),
rh_mw (clear-sky mean radiant heat, clear no-flare passes count 0),
bcm_per_year, rh_max_mw, temp_k, n_overpasses, n_clear, n_detected,
n_detected_cloud, n_clear_detclear, rh_mw_detclear, bcm_per_year_detclear.

`overpasses` (one row per site per granule pass): satellite (SNPP | NOAA-20 |
NOAA-21), granule, time_utc, site_id, row, col (scan position; col 0-3199,
edges are high view angle), dist_m, sza, cloud_mask (0/1 clear, 2/3 cloudy),
cloud_clear_neighbors, detected, rh_mw, rh_peak_mw, temp_k, area_m2, peak_lat,
peak_lon, peak_dist_m, fit_flag, and more. Use `pipeline.dedupe_overpasses`
(collapses granule-edge duplicates) and `pipeline.strict_clear` (the clear-sky
rule used for nightly values). A local night uses passes with solar zenith
>= 97 deg; local night = (UTC time + lon/15 h - 12 h).date().

`copies` (one row per detected hot source copy): centroid_lat/lon,
assigned_site, site_dist_m, group_rh_mw, temp_k, fit_flag, etc.

Method and limits: `nightly-flaring/docs/explainer.md` (plain language, with
the limits section) and `nightly-flaring/docs/decisions.md` (every method
choice, D01-D57). Known data issues: night 08-09 is partial (4,223 sites not
observed); SNPP has no passes after 2 Nov 12:51 UTC; non-upstream flares are
not flagged; one pass is a sample, nights are 1-3 passes.

## Rules

- Use only the `nightly-flaring/` folder: its data, code and docs. Do not use
  or read other work in the repository (e.g. `offshore_methane/`, `notebooks/`).
- Read-only on code and docs. Write only inside
  `nightly-flaring/outputs/flareout_research/` (memos, and scripts under
  `scripts_sol/` or `scripts_opus/`). No commits, pushes or installs.
- Web search for literature is fine (unlit flares, venting, flare outages,
  offshore super-emitters, change-point methods); cite what you use.
- No secrets are needed; do not print any.
- Writing: state claims plainly; put uncertainty in numbers and n
  (`50% (8/16)`, not "suggestive"); no filler such as "importantly",
  "notably", "worth noting", "to be clear", "honestly"; no sentences that rate
  your own work. Use the weakest verb the evidence supports
  (is consistent with < suggests < shows).
- Keep each memo under ~1,500 words plus tables. Name files exactly as asked.

## Rounds

- Round 1: independent. Write `<you>_r1.md`: 3-6 ideas in the format above,
  then "Questions for my partner" (what you want them to check or challenge).
- Round 2: read your partner's `*_r1.md`. Write `<you>_r2.md`: assess each of
  their ideas (keep / merge / drop, with reasons; run a check where the dispute
  is empirical), revise your own, and propose a joint shortlist of at most 4
  ideas, ranked, each with the conditions under which you would trust it.
- Round 3: read your partner's `*_r2.md`. Write `<you>_r3.md`: for each item on
  both shortlists, ENDORSE / ENDORSE WITH CHANGES (say which) / REJECT, plus the
  remaining disagreements.
