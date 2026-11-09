# Sol — Round 3: source-level thermal loss, with conditional interpretation

Surface a thermal-departure workflow and an inspection queue. The nightly data
cannot establish continuing production, physical extinction, or methane venting.
4582/4583 is the leading recently bright disappearance; 4497 does not establish
a simultaneous independent stop. A newly checked heat reduction at 4500 adds
local context even though that source remained detected nightly.

I re-read the brief, both Sol memos/notes, Opus's Round 2/notes, and s09–s10.
New checks use the supplied pass cache and **original copy CSVs**, deduplicated
by `satellite, granule, group_id`, including cloudy/unassigned positive evidence.
No annual catalogue, external data, installs, downloads, or project edits were used.

## Decisions on both shortlists

Corresponding entries below are one idea, not separate recommendations.

| Shortlist entries | Decision | Exact changes and reason |
|---|---|---|
| **Sol 1 ↔ Opus 1:** recently bright absence screen, source tracking/veto | **ENDORSE WITH CHANGES** | Keep opportunity-night ranking and source reassignment checks. Replace universal “10 nights, or 6 with wider pooling” with a prespecified, partially pooled duration model and fleet-scan calibration. Call 4582 an **inspection priority**, not today's calibrated alert. Its prior **seven-night** source median is **12.13 MW (25 groups)**; 14.62 MW (45) describes fourteen nights. Freeze source centres/eligibility before evaluation; a veto preserves only a valid underlying screen's error bound. |
| **Sol 2 ↔ Opus 3:** censored heat/state-duration model | **ENDORSE WITH CHANGES** | Use **normal heat / reduced heat / unresolved darkness**. An internal “off” state may describe the model's observation regime; never report its posterior as the probability of physical extinction. Remove the universal 29-pass sufficiency rule and fixed 0.3-MW floor. Weak nearby heat prevents a complete-absence claim but does not establish the same physical flame; overlapping localization distributions require abstention. |
| **Sol 3 ↔ Opus 2:** independent-source common-mode context | **ENDORSE WITH CHANGES** | Add the ≤10-km layer to 50/100-km context, exclude the target's attribution complex, and compare **heat changes as well as detection occupancy**. Replace causal “shared stop” labels with observed local/regional thermal-change descriptions. The 4582/4583 and 4497/4485 platform pairs cannot each supply two independent donor histories. |
| **Opus 4:** small-flare/cloudy-region watch tier | **ENDORSE WITH CHANGES** | Keep a review queue within the same workflow. Separate weak-flare ambiguity (3662/3644), a short observed episode (6011), and a recently bright but short dark interval (927). Lit/observed-night eligibility can admit cloudy regions; it does not estimate clear-opportunity detection probability. Keep unknown/cloudy nondetections out of negative exposure, refit the family, and make no flare-failure claim. |

## (a) Clustered misses: working Markov model, block checks, no certified null

Independent nights are no longer a disagreement: **both Round 2 memos reject
their calibration**. Sol's baseline has seven adjacent qualified misses versus
1.65 under within-site exchangeability; 6/20,000 permutations reach seven.
This does not identify whether clustering comes from observation errors or
real thermal changes.

Let Q mean ≥2 raw-cloud-0/1 passes; H means any detection, M a qualified miss.
My new audit reproduces **P(M|M)=7/19** at the 21 anchors. Opus's
P(M|H)=12/549 includes 63 extra H→H transitions from non-Q positive nights;
the Q-only denominator gives **12/486**. Non-Q detections should reset a gap,
without becoming extra clear-opportunity trials. Explicitly respecting those
resets gives 7/18 miss-persistence trials. The Q-only Beta(8,13) posterior's
95% interval for b is **0.19–0.59**.

A first-order model is a starting specification:

\[
a_{it}=P(M_t\mid H_{t-1},x_t),\quad
b_{it}=P(M_t\mid M_{t-1},x_t),\quad
T_{it}(k)=E\!\left[a_{it}\prod_{j=1}^{k-1}b_{i,t+j}\mid\text{baseline}\right].
\]

Pool by observation quality, recent heat and source, retaining site uncertainty;
include calendar spacing and satellite/pass composition in `x`.
Site-pixel cloud flags remain imperfect proxies at learned source centres.
Use a separate boundary calculation when no preceding positive is observed.
The diagnostic below uses constant transitions with independent Beta(1,1) priors,
site-specific `a`, and pooled `b`. Entries are **1,218×E[a b¹³]** for 4582,
not validated p-values or probabilities of venting.

| Observation-unit/transition sensitivity | Pooled b data | Adjusted model score |
|---|---:|---:|
| Q-only site histories | 7/19 | 0.0206 |
| Q trials with non-Q positive resets | 7/18 | 0.0340 |
| Fixed 1-km source-aperture proxy at the same 21 anchors | 11/20 | 0.262 |

The aperture proxy has localization failures and is **not** a requalified source
cohort; its result demonstrates the need to recalibrate a source-level refit.
Even on site histories, pooling `a` across anchors instead changes 0.0206 to
0.00439. Opus's observed-night eligibility produces **25**, not 21, bright
site histories: changing the cohort also changes the scan family. The wider
70-site reference has b=12/76; choosing that pool because it shortens the
required gap is unsupported.

Use first-order Markov as the working baseline, adding duration structure if
blocked prediction retains residual memory. Use calendar blocks to check
residual duration and cross-source dependence, not to declare a benign null.
New 20,000-replicate baseline resampling gives
0.00005/0.0036/0 mean events under the old independent threshold for
1/3/7-night blocks. The longest baseline miss run is only three opportunities;
just **two** transitions follow a third miss. Seven-night resampling cannot
establish the fourteen-night tail. Conversely,
Round 2's full-record block shuffles preserve actual operating changes.
Neither supplies measured false-alarm rates. Calibrate the maximum over sources
and starts after fixing eligibility; the proposed ≤0.1 false thermal alerts per
30-night fleet scan remains a **model-based target**, not an achieved guarantee.

## (b) What a fitted “off” state can mean

Report the observed opportunity count, heat history, and a conditional model-state
probability with sensitivity to low-flame detectability and false detections.
For zero detections, an illustrative independent-pass likelihood ratio is

\[
BF_{O:W}=\left(\frac{1-\epsilon}{1-p_W}\right)^n.
\]

At ε=0.002 and **45 misses**, BF is **1,371** for p_W=0.15,
**1.44** for p_W=0.01, and **1.0045** for p_W=0.0021. These are alternative
assumptions about an unobserved low flame, not estimates from labelled states;
correlated passes further invalidate a universal pass-count rule.
As p_W approaches ε, darkness and a sufficiently small continuing flame are
observationally equivalent. Opus's “no thermal evidence” wording resolves this
if retained consistently. Erasure/dimming injections measure performance under
their assumed emissions; they cannot validate physical extinction probabilities.

## (c) 4582, 4583 and 4497: identity versus co-change

**4582/4583: one attribution complex for review, not two independent events.**
Their baseline centres are 498 m apart; 94 baseline passes contain no two
≥3-MW groups. This does not prove one physical flame.
**4497 is a separate resolved source:** its centre is 8.60 km away, and
84/105 baseline passes contain bright groups in both non-overlapping apertures.

| Source history | Direct copy-group evidence |
|---|---|
| 4582 complex | Oct 21–Nov 3: zero groups within 3 km; 45 raw-clear site passes across three satellites (44 with ≥6 clear neighbours). A 0.190-MW group, 626 m from its baseline centre, appears Nov 4. |
| 4497 complex | Major heat change around Sep 13: adjacent fourteen-night clear medians **26.54→1.41 MW (48/12 groups)**. During Oct 21–Nov 3 it remains detected on **7/14 nights**. Oct 23/24/30 sources are assigned to 4485; all remain within 750 m. The source-level October gap is Oct 25–29, not a co-stop within two nights of 4582. |
| 4500 source, 6.66 km from 4582's source | Clear medians **11.19 MW (47 groups, Oct 8–21)→0.550 (35, Oct 22–Nov 1)→76.38 (7, Nov 2–4)**. It remains detected nightly. Nadir/≥6-clear-neighbour medians also fall **10.97→0.505 MW (14/12)**; all three satellites and 0.5–1.5-km apertures retain the reduction. |

Thus 4582 and 4497 have distinct major-loss dates; neither their statistical
independence nor a shared cause is established. **4500 supplies a nearby
coincident thermal reduction**, so I also withdraw an unqualified “isolated
loss” description of 4582. This supports heat-aware local context, without
identifying a common gas system, maintenance, or shutdown.

Reproduction: [`scripts_sol/README.md`](scripts_sol/README.md), Round 3 commands.
The Markov/heat-state production models remain proposals; the numerical audits
and source/quality sensitivities above were executed.

## Final agreed list (my side)

1. **Source-centred, recently bright thermal-departure screen.** Rank sustained
   absences after stable, bright history using clear opportunities, source
   reassignment checks and duration-aware uncertainty. Surface the 4582/4583
   complex as an inspection priority with its evidence card and residual-heat return.
2. **Censored heat-state and change-duration model.** Date major heat reductions
   before interpreting later nondetection runs, separating reduced thermal emission
   from unresolved darkness. Validate blocked prediction and synthetic dimming,
   erasure and assignment perturbations, while reporting assumptions about unseen flames.
3. **Independent-source context across geographic scales.** Compare heat and
   occupancy at ≤10, 50 and 100 km after resolving attribution complexes.
   The 4500 co-change prevents treating nearby continuing detections as proof
   that 4582's operating context stayed unchanged.
4. **A watch queue for limited or ambiguous evidence.** Prioritize 927's recent
   bright history and six dark nights for inspection, while retaining 3662/3644
   as weak-source absences and 6011 as a short observed episode. These are review
   leads; the record does not establish that they should still have been flaring.

## Remaining disagreements

- **Alert status/calibration:** I retain evidence cards until the observation unit,
  eligibility, pooling and duration calibration are fixed; Opus's “one alert today”
  is stronger than the current audit supports.
- **State semantics:** no remaining disagreement if “off” means unresolved model
  darkness; physical-extinction posteriors and a universal 29-pass rule are rejected.
- **Cause:** the 4497 co-stop claim is not supported after source correction.
  Whether the 4582/4500 changes share a cause remains unidentifiable here.
