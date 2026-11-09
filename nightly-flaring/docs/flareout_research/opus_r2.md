# Opus round 2: assessment, revisions, joint shortlist

New scripts: `scripts_opus/s09_r2_checks.py`, `s10_entity_anchors.py`
(run from `nightly-flaring/`). Sol's definitions are used where stated:
baseline 08-10 to 09-07; a qualified night (Q) has ≥ 2 deduplicated passes
with `cloud_mask` 0/1; any detection, clear or cloudy, makes a night lit.

## Checks on the disputed points

| Check | Result |
|---|---|
| Serial dependence at Sol's 21 anchors, baseline Q-night transitions | P(miss \| previous miss) 0.37 (7/19) vs P(miss \| previous lit) 0.022 (12/549); marginal miss rate 0.033 |
| Same, 70 sites with ≥ 12 Q nights and lit ≥ 0.8 in baseline (any size) | 0.16 (12/76) vs 0.038 (64/1,683) |
| Same-night misses inside lit spells (previous and next informative nights lit) | Sol anchors: 35 all-miss nights vs 59 expected under independence (n = 1,327); my 126 steady sites: 196 vs 133 (n = 4,649) |
| Outcome-dependent clearing in `strict_clear` | 233 of 24,239 clear detections are cloud-flagged (1.0%); at anchors 19 of 4,072 |
| Entity-level eligibility (DBSCAN 750 m, r95 ≤ 1.5 km) | ≥ 3 MW: 18 entities (12 span > 1 site) vs 21 site anchors; ≥ 1 MW: 31; ≥ 0.5 MW: 54. Entity r95 median 677 m; 10 of 405 entities chain beyond 1.5 km |
| Site 927 (Gulf of Mexico, 6.3 MW) | Baseline: 6 Q nights (6 lit), lit on 26 of 29 nights; cloud-flagged passes detected 63% (116/183). From 10-30: 6 Q nights, 15 clear passes, 2 cloudy passes, no detection |

Sol's five nominal gaps under a first-order Markov null, P(miss | miss)
integrated over its Beta(8, 13) posterior from the anchors' baseline,
P(miss | lit) = 0.022, scaled by M = 1,218:

| Gap | k | Sol (independent nights) M×T | Markov M×P |
|---|---|---|---|
| 4582 Oct 21–Nov 4 | 15 | 1.0e-5 | 2.2e-3 |
| 3669 Oct 23–Nov 4 | 10 | 2.6e-3 | 0.030 |
| 4497 Oct 23–30 | 8 | 5.2e-5 | 0.10 |
| 4286 Oct 3–6 | 4 | 0.039 | 1.8 |
| 4497 Oct 15–18 | 4 | 0.039 | 1.8 |

The minimum gap reaching M×P ≤ 0.05 rises from 4 Q nights (Sol's
`required_qnights.csv`, 12 of 21 anchors) to 10 Q nights under the
anchors' Markov posterior, or 6 under the 70-site estimate.

Neighbourhood of 4582 (sites within 10 km with a detection 10-08 to 11-04,
nightly status from 10-08; F flaring, f lit under cloud, . clear no flare,
c cloudy):

| Site | km | 10-08 → 11-04 |
|---|---|---|
| 4582 | 0 | `FFFFFfFFFFFFF...............` |
| 4583 | 2.4 | `FFFFffFFFFFFF..............F` |
| 4497 | 7.5 | `FF...cf....FFFF........F...F` |
| 4500 | 5.7 | `FFFFFfFFFFFFFFFFFFFFFFFFFFFF` |
| 4485 | 9.5 | `FFFFFc.FFFF..FFFF.....F.....` |

Seven further sites within 10 km had 1–5 scattered detections.

## Sol's ideas

**1. Opportunity-time departure from a persistent baseline: KEEP, merged
with my idea 1.** Night-level scoring is the better unit: at my 126 steady
sites, same-night misses run 1.47× the independence expectation (196 vs
133), so my pass-level CUSUM overstated evidence at small flares. At bright
anchors there is no excess (0.59×). Two changes. (a) The independent-night
null fails: misses cluster (0.37 vs 0.022), and under a Markov null 3 of
the 5 gaps lose significance. (b) Requiring ≥ 12 Q nights in a 29-night
baseline excludes cloudy regions by construction (no Gulf of Mexico
anchor); 927 was lit on 26 of 29 baseline nights with only 6 Q nights.
Eligibility can use lit / observed nights, counting cloudy non-detection
nights as misses; this denominator does not depend on the outcome and
understates q.

**2. Source-centred tracking: KEEP as a veto, merged with my idea 3.** The
aperture veto removes 4286 and the earlier 4497 gap, both of which also
fail the Markov null. Source-centred units do not widen bright coverage
(18 entities vs 21 sites at ≥ 3 MW), so I drop my claim that entities
expand the monitored set; their use is attribution and vetoes.

**3. Local exceptions to shared shutdowns: KEEP WITH CHANGES.** Donors at
50–250 km do not test a shared operating change at the sub-field scale.
Of the four other sites within 10 km of 4582 that were lit on most
nights before 10-21, 4583 and 4497 went dark within 2 nights of it, 4485
was lit on 1 of the last 11 nights, and 4500 stayed lit 28 of 28 nights. The gate needs a ≤ 10 km layer, and a
co-stop there should label the event "sub-field stop" rather than clear or
condemn it. My co-stop count (9 of 84 runs vs 4.1 expected, 95th
percentile 8) supports the same layer.

**4. Major heat loss vs disappearance below sensitivity: KEEP, merged with
my idea 2.** Sol's 3669 (5.98 → 0.32 MW on Aug 29, 45 / 16 source groups)
and my 4497 (26.7 → 1.09 MW, per-pass detection 0.82 → 0.18 at nadir)
show the same pattern. The AR(1) log-heat term belongs in my lit / low /
off model. Sol's pre-gap gate (≥ 3 MW over the prior week) is the
condition under which a dark gap is informative.

## Answers to Sol's questions

1. **Source-centred eligibility.** Not for bright flares: 18 entities vs
   21 sites at ≥ 3 MW. Coverage grows only by lowering the heat threshold
   (31 at ≥ 1 MW, 54 at ≥ 0.5 MW), which imports the turndown problem.
   Entities do help cloudy coverage if eligibility counts lit / observed
   nights (927).
2. **Weak residual source.** A separate state, not a veto: "major heat
   loss with residual flame" (state W) differs from "no thermal evidence"
   (state O). Internal validation: inject turndowns (real detections
   thinned by the size-dependent p from round 1: 0.81 nadir at < 1 MW vs
   0.89 at > 10 MW) and erasures into anchor histories, and report the
   W-vs-O confusion matrix; require the residual source's centroid inside
   the baseline r95 (531–743 m in Sol's data) to rule out a neighbour.
3. **Serial-dependence null with 29 baseline nights.** A first-order Markov
   chain on qualified nights, with P(miss | miss) pooled hierarchically
   across a reference set wider than the anchors (70 sites: 0.16, 12/76)
   and integrated over its posterior. Yes, it invalidates 3 of the 5
   alerts (table above). Block-bootstrap of whole weeks is the check.
4. **Donors for 4582.** Not credible at 50–250 km. At ≤ 10 km, 2 of 4
   other regularly active sites co-stopped and 1 stayed lit every night.
   A change in a gas system shared by 4582 and 4583 would pass both of
   Sol's screens.

## Revisions to my ideas

- Idea 1 becomes a night-level score with a Markov null; my 84-run count
  (pass-level, stationary null) overstates alerts.
- Idea 2 adds Sol's AR(1) log heat and a prior-week ≥ 3 MW condition for
  alert tier.
- Idea 3 is a veto and attribution layer, not a coverage expander.
- Idea 4 adds a ≤ 10 km co-stop layer.
- New: cloud-aware eligibility (lit / observed nights) so cloudy regions
  can enter a watch tier.

## Joint shortlist (ranked)

**1. Bright-anchor absence screen with source-aperture veto** (Sol 1 + 2,
Opus 1 + 3). Night-level score, Markov null, Bonferroni over site-nights.
*Trust when:* ≥ 12 observed baseline nights with lit ≥ 0.8; prior-week
median heat ≥ 3 MW; gap ≥ 10 Q nights (6 if P(miss | miss) is pooled over
the wider reference set); clear misses from ≥ 2 satellites; no source
within 1.5 km of the baseline centroid, checked to 3 km. *Residual
failure:* a benign single-platform stop (maintenance, gas recovery,
shut-in) looks identical. Today one alert: 4582, Oct 21–Nov 3 (14 Q
nights, Markov M×P 3.6e-3; prior-week 14.6 MW; all three satellites). A
0.19-MW source 626 m from its baseline centre on Nov 4 (assigned to 4583)
ends the gap and is a W-state candidate for item 3. 3669 (0.030) fails the
prior-week heat condition (0.36 MW) and moves to item 4.

**2. Multi-scale shared-stop labels** (Sol 3, Opus 4). Each alert gets
isolated / sub-field (≤ 10 km) / regional, from donor activity at 10, 50
and 100 km over the gap nights. *Trust when:* ≥ 2 active donors at the
scale and ≥ 3 donor components at 100 km observed on most gap nights.
*Residual failure:* a stop confined to the target's own gas system; 4582
and 4583 may share one.

**3. Heat-state model: bright / dim / dark** (Sol 4, Opus 2). Separates
"no thermal evidence" from "residual flame". *Trust when:* pre-gap median
≥ 3 MW; ≥ 29 clear passes (about 15 clear nights) for 100:1 between off
and a p = 0.15 low state; W-vs-O confusion measured on injections.
*Residual failure:* a flame below about 0.3 MW is indistinguishable from
none.

**4. Watch tier for small and cloudy-region flares** (Opus 2 + cloud-aware
eligibility). Reports "stopped or below floor", never an alert. Current
entries: 3662, 3644 (Persian Gulf, < 1 MW, dark about 47–50 nights),
6011 (Bohai), 927 (Gulf of Mexico, 6.3 MW, 6 dark Q nights). *Trust
only* as a prompt for more nights or imagery. *Residual failure:* turndowns;
35 of 74 closed dark runs returned at under half their earlier heat.

## Main disagreement

The null. Sol's independent-night Beta null treats each qualified miss as
fresh evidence; the anchors' own baseline shows a miss after a miss 17
times as often as after a lit night (0.37 vs 0.022, n = 19 and 549). Under
that dependence only gaps of about 10 Q nights or longer clear a
Bonferroni 0.05, and 4582 is the one alert that holds by a wide margin.
