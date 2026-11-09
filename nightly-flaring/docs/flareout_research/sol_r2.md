# Sol — Round 2: a thermal-departure model with an abstention rule

Build a source-centred thermal-loss screen, then use heat history and common-mode
activity to interpret it. None of these models identifies continuing production,
physical extinction, or methane venting from this dataset alone.

I re-read both memos and scripts. Checks use the two v5 segments: cached passes
and original copy CSVs, collapsed by satellite/granule/group_id. A raw-file
audit matched 3,548 pass records at 11 sites but found 228 source groups omitted
by Opus's component/time copy deduplication (39,314 original versus 39,086 cached,
before night trimming). Spatial results below use the original groups.
No annual catalogue, external data, downloads, installs, or project-code edits
were used. Reproduction: [scripts_sol/README.md](scripts_sol/README.md).

## Assessment of Opus's four ideas

**1. Dark-run CUSUM — MERGE with my opportunity-night screen; retain ranking,
replace its calibration.** I reproduce all three thresholds. The product
exp(−S) describes a particular pass string under an independent, known-probability
null; estimated probabilities and repeated fleet-wide scanning require additional
calibration. Changing strict-clear to outcome-independent cloud flags 0/1 leaves
84 runs at the middle threshold, using the same offsets.

| Check at S ≥ ln(10,000) | Mean runs | Replicates |
|---|---:|---:|
| Observed histories | 84 at 51 sites | 1 dataset |
| Opus within-site/view-zone pass shuffle | 1.60 | 20 |
| Shuffle within site × zone × satellite × neighbour-clear category | 1.96 | 100 |
| Shuffle whole calendar-night bundles | 34.45 | 100 |
| Shuffle contiguous three-night bundles | 77.55 | 100 |
| Shuffle contiguous seven-night bundles | 93.30 | 100 |

Bundles carry outcomes, geometry, cloudy detections, and unknowns together.
They preserve real operating changes too: these are dependence stress tests,
**not measured false-alarm rates**. The contrast cannot isolate cloud persistence
from operating persistence, but finer geometry stratification alone does not
resolve it.

My independent-night shortcut also fails a baseline check: 526 qualified nights
at 21 anchors contain 20 misses and seven adjacent miss pairs, versus 1.65
expected after within-site permutation; 6/20,000 permutations reach seven.
The pooled miss-following-miss rate is 7/19, versus 12/486 after a lit night.
Under a stationary Markov stress model with each site's posterior miss fraction
πᵢ, a k-miss tail is πᵢr^(k−1). At r=7/19, three of my five original runs
retain nominal adjusted tails ≤0.05; at r=0.592, none do. That upper value is the
97.5th percentile of Beta(8,13), not a validated benign-state bound.

Applying my unchanged thresholds to 20,000 simulated 58-night monitoring periods,
omitting nonqualified-night positive resets, gives 0.022 alerts/period under
independent nights and 0.776 under r=7/19.
Neither simulation estimates false methane accusations. Both screens need
duration-aware, source-level calibration before their scores become alert
probabilities. On the same 526 baseline opportunities, the 1-km source-aperture
proxy has 11 adjacent miss pairs versus 4.57 expected under within-source
exchangeability. Source correction alone does not remove clustering; hard
apertures can introduce localization misses too.

**2. Lit/low/off semi-Markov model — MERGE with my heat-loss model; DROP literal
extinction probabilities as an identified output.** Keep duration pooling and
censored heat likelihoods. Call the states normal heat, reduced heat, and
unresolved darkness. A low-state constraint p_W≤0.3 allows p_W arbitrarily close
to the off-state rate; the data then cannot distinguish them. With ε=0.002,
BF=((1−ε)/(1−p_W))^n reaches 100 after 29 clear misses at p_W=0.15,
94 at 0.05, and 573 at 0.01. For 927's 15 clear misses, those assumptions give
BFs 11.1, 2.09, and 1.13. A large off:low BF would depend on excluding very
small continuing flames through the prior. Estimate observation parameters with
held-out passes; assess event calibration with whole-night/time-block holdouts
and erasure, dimming, and assignment-swap injections.

**3. Flare-centred entities — KEEP; use DBSCAN as a proposal generator.**
Baseline-only clustering of 12,619 copy groups gives 329 entities, with 480
groups classed as noise; 39 entities exceed 2 km diameter. A strict 2-km
diameter/separation guard leaves only ten eligible two-clear-pass entities,
versus my 21 site anchors, and excludes even isolated 927 because its full scatter diameter
is 2.24 km. Source identity improves validity; automatic coverage expansion is
not established. Learn scan-dependent location uncertainty, reject unresolved
overlaps, and test radius sensitivity. Site-pixel clearing remains an imperfect
proxy for clearing at the learned source.

**4. Regional/field common-mode terms — KEEP; DROP “likely shut-in” as a causal
label.** Shared operational loss, shared flare failure, and atmospheric errors
can produce the same term. Exclude the target's entire attribution complex from
its donors. For 4582/4583, platform separation is 2.36 km but baseline source
centres are only 498 m apart: 59/62 and 29/31 assigned groups fall within 1 km of
the other centre. Across 94 nearby baseline passes, none has two groups ≥3 MW;
three have a weaker second group. Their apparent co-stop is not independent
evidence of two bright sources stopping. Also, 4497 was already dim before its
October gap, so coincident run starts need not be coincident heat-loss onsets.

## Answers to your questions

**1. Reproduction and null:** see the table. One cohort correction: the scan
considers 421 sites with ≥6 clear detections and applies eligibility before each
run. Eighteen of the 84 flags, at 13 sites, fall outside the retrospectively
defined 221-site monitored cohort. Restricting to those 221 gives 66 flags at
38 sites. The denominator matters for the alert budget.

**2. ε:** no labelled-unlit exposure exists here, so I cannot estimate a physical
unlit false-detection rate. Illustrative raw-cloud-flag calculations:

| Selection | Sites | Clear detections / clear passes | Rate |
|---|---:|---:|---:|
| ≤1 clear detection over the full 88 nights | 7,084 | 98 / 1,192,041 | 0.0000822 |
| No detection of any kind during Aug 10–Sep 7; evaluated later | 7,099 | 955 / 793,535 | 0.00120 |

The first selection forces the numerator small. The second includes new genuine
flaring and attribution changes: 195 previously quiet sites later have a
detection. Neither is an off-state truth set. Sweep ε as a sensitivity parameter.

**3. Candidate checks:** all four have **zero recorded copy groups within 4 km**
of their baseline source centre during the pass-level gap. Counts use grouped
sources, including unassigned and cloudy detections.

| Site | First fully dark local night | Clear misses, through Nov 4 | Prior-week clear source heat, median MW (groups) | Closest recorded source during pass gap |
|---|---|---:|---:|---:|
| 3662 | Sep 19 | 148 | 0.95 (20) | 23.5 km |
| 3644 | Sep 16 | 159 | 0.56 (14) | 18.2 km |
| 6011 | Sep 16 | 126 | 0.51 (23) | 15.2 km |
| 927 | Oct 30 | 15 | 9.44 (11) | 6.74 km |

The first three are sustained thermal absences, with a weak-flame alternative.
6011's entire detected episode spans only Aug 29–Sep 15, on 16 nights.
927 warrants higher-priority inspection: its 15 misses span all three satellites,
13 have ≥6 clear neighbours, and six are nadir passes. Oct 29 still has two
cloudy detections; the fully dark interval is six nights, not seven.

**4. Bright-flare misses:** reproduced 94/820 nadir misses at 12 selected >10-MW
sites. Even eight clear neighbours leave 90/753 misses. For 38/94 misses, a
source lies within 1 km of the learned baseline centre in the same satellite
pass, consistent with attribution contributing. This does not resolve every
source's identity. All 1,260,494 clear nondetections have n_hot_m10=0:
`granule.site_cluster_summary` counts hot pixels only in confirmed assigned
groups. It contains no rejected-trigger diagnostic. Remaining cloud errors,
flame variability, and pipeline rejection cannot be apportioned from these
tables.

**5. Gulf of Mexico:** an early-October common reduction exists in the 19-site
monitored subset. Clear-pass detections are 241/363 during Sep 21–Oct 4,
42/95 during Oct 5–11, and 98/162 during Oct 12–18. A retrospective
site/zone/satellite/neighbour-clear logistic model leaves a −0.195 mean residual
during Oct 5–11. Yet 927 remains detected on 10/11 nights during Oct 5–15.
Call this an unattributed common reduction, not a demonstrated storm shut-in.

**6. Budget:** provisionally target ≤0.1 model-based false thermal-departure
alerts per 30-night fleet scan, and ≤5 distinct source episodes/month for
analyst review. Calibrate the maximum over sources and scan starts, not a
per-run α. These are proposed operating targets. With 88 nights and no labelled
benign/venting outcomes, actual accusation error is unmeasurable.

## Revisions and ranked joint shortlist

My two-clear-pass eligibility was too restrictive for cloudy regions. Allowing
one clear pass, retaining ≥12 baseline qualified nights, expands 21 anchors to
24 and includes 927 (12/12 lit qualifying nights versus six two-pass nights).
Its six-miss adjusted independent-night tail is **0.0513** across that full
24-site family: an inspection priority, not a nominal 0.05 crossing. Refitting
source units or eligibility requires fresh calibration; it does not inherit a
veto-only error bound. Weak residual heat blocks a complete-absence claim but
retains a major-heat-loss hypothesis, including 4582's 0.19-MW Nov 4 source.

1. **Source-centred, recently bright thermal-departure screen.** Merge our run
   methods with source tracking and duration calibration. Trust an absence
   ranking when localization is stable, recent heat was well above sensitivity,
   multiple satellites supply clear opportunities, and block/Markov sensitivity
   meets the fleet budget. Until then, provide evidence cards rather than
   calibrated alerts. **Main residual failure:** an individual benign production
   stop or gas diversion has the same thermal signature.
2. **Censored heat-state and change-duration model.** Merge our low/bright/dark
   ideas; date the heat change before testing an unusual spell. Trust reduced-heat
   outputs only after geometry-aware blocked prediction and injected dimming
   checks; require robustness to low-flame priors for any darkness comparison.
   **Main residual failure:** an arbitrarily small flame and zero flame remain
   observationally indistinguishable.
3. **Independent-source common-mode control.** Retain regional and local terms
   as contextual ranking/suppression, after entity checks. Trust common-loss
   descriptions with several independent donor tracks and stable results across
   geographic scales and observation-quality definitions. **Main residual
   failure:** shared failures can resemble benign co-stops, while isolated
   maintenance escapes the common term.

My largest disagreement is with treating a fitted “off” posterior as physical
extinction evidence; the state distinction depends on unidentifiable low-flame
and false-detection assumptions.
