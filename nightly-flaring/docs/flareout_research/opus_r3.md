# Opus round 3: verdicts, open disagreements, final list

New scripts: `scripts_opus/s11_r3_checks.py`, `s12_r3_bounds.py` (run from
`nightly-flaring/`). Sol's audit is correct: my copy cache deduplicated on
component and time and dropped 228 source groups. Only my round-1 "hot
source within 2 / 2–4 km" counts used it; Sol's re-run on the original
groups finds zero sources within 4 km during the gaps at 3662, 3644, 6011
and 927, so no candidate changes.

## Checks

| Question | Result |
|---|---|
| Is miss clustering geometric? Sol anchors, baseline Q nights | 7 adjacent miss pairs vs 1.41 expected (within-site shuffle) and 1.59 (within site × night geometry: best clear zone, 2 vs ≥ 3 passes); P < 0.0005, 2,000 shuffles |
| Same, 70-site wider set | 12 vs 6.40 and 6.76; P = 0.02 and 0.03 |
| Is it shared between sites? Pairs 3–50 km apart, same Q night, wider set | Co-misses 10 vs 3.3 expected in the baseline (75 pairs, 1,695 shared nights); 160 vs 106 over all nights (5,250) |
| Re-detection on the same night after one clear detection at RH | ≤ 0.2 MW: 0.19 (n = 934); 0.2–0.3: 0.34 (3,071); 0.3–0.5: 0.53 (7,227); 0.5–1: 0.71 (11,707); 1–5: 0.79–0.81; > 10: 0.76 (8,922) |
| 4497 lit × 4582/4583 lit, nights both observed, 09-08 to 11-04 | odds ratio 1.87, Fisher P = 0.53 (n = 57) |

Night-level heat bounds: probability of the observed dark nights if a flame
like those detected in each RH class had persisted, one pass per night with
≥ 1 clear pass, nights independent:

| Spell | Nights | ≤ 0.2 MW class | 0.2–0.3 | 0.3–0.5 | Lowest class excluded at 0.05 |
|---|---|---|---|---|---|
| 4582, 10-21 to 11-03 | 14 | 0.052 | 2.9e-3 | 2.4e-5 | 0.2 MW |
| 927, 10-30 to 11-04 | 6 | 0.28 | 0.082 | 0.011 | 0.3 MW |
| 3669, 10-23 to 11-04 | 10 | 0.12 | 0.015 | 5.0e-4 | 0.2 MW |
| 3662, 09-19 to 11-04 | 44 | 9.4e-5 | 1.1e-8 | < 1e-14 | all classes |
| 3644, 09-16 to 11-04 | 45 | 7.6e-5 | 7.1e-9 | < 1e-14 | all classes |
| 6011, 09-16 to 11-04 | 35 | 6.3e-4 | 4.6e-7 | < 1e-11 | all classes |

Nightly median clear RH (MW) around the 4582 stop; blank = no clear
detection:

| Night | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 → 11-01 | 11-02 | 11-03 | 11-04 |
|---|---|---|---|---|---|---|---|---|---|
| 4582/4583 complex | 12.4 | 19.3 | 6.8 | | | | | | 0.2 |
| 4500 (5.7 km) | 17.9 | 54.5 | 40.5 | 59.6 | 0.7 | 0.5–0.9 | 87.3 | 72.7 | 79.8 |
| 4497 (7.5 km) | | 0.2 | 0.3 | 1.5 | 0.6 | 0.4 on 10-31 | | | 0.2 |

## Verdicts (corresponding items merged)

**A. Bright-flare thermal-departure screen** (Opus 1 = Sol 1).
ENDORSE WITH CHANGES:
1. Output is an evidence card, not an alert, until calibration meets
   Sol's budget (≤ 0.1 model-based false departures per 30-night fleet
   scan). Rank by the Markov tail at r = 7/19; label "departure" only if
   the tail holds at r = 0.59 (97.5th percentile of Beta(8, 13)).
2. Sol's one-clear-pass eligibility (24 anchors, 927 included).
3. Unit = source complex: 4582 and 4583 are one source (centres 498 m apart;
   no pass with two groups ≥ 3 MW in 94 baseline passes, Sol).
4. Every card carries the night-level heat bound above and the item C
   context.

**B. Heat-state model** (Opus 3 = Sol 2). ENDORSE WITH CHANGES: Sol's state
names (normal heat / reduced heat / unresolved darkness); date the heat
change before scoring the spell (3669 fell 5.98 → 0.32 MW on Aug 29; 4497
26.7 → 1.09 MW around Sep 13). Darkness is reported as the heat bound in
the table, never as an off-state probability. I withdraw my O:W Bayes
factor (100:1 after 29 misses): it holds only for p_W = 0.15 and needs 573
misses at p_W = 0.01 (Sol).

**C. Common-mode context** (Opus 2 = Sol 3). ENDORSE WITH CHANGES:
1. Donors are scored on heat (log RH against their own baseline), not on
   lit status. 4500 was lit on 28 of 28 nights in my round-2 table yet fell
   from 40–60 MW to 0.5–0.9 MW on 10-22, the night after the 4582 complex
   went dark.
2. Exclude the target's attribution complex from its donors (Sol).
3. Scales 10, 50, 100 km; label "shared heat loss" or "isolated", never
   "likely shut-in" (Sol).

**D. Watch tier for small and cloudy-region flares** (Opus 4; no Sol
item). ENDORSE WITH CHANGES: fold into A as a second card type, "sustained
darkness at a small flare", with the heat bound. 927 moves to A through
the one-clear-pass rule. 3662, 3644 and 6011 stay here: every detected
size class is excluded (P ≤ 6.3e-4), but their pre-gap heat was 0.5–0.95
MW (Sol), so a flame below the smallest detected class remains possible.

**REJECT** (my earlier claims): entities as a coverage expander (18 vs 21
anchors, and Sol's 2-km guard leaves 10); my 84-run count as an alert
budget (pass-level, stationary null).

## Open disagreements

**(a) Null for clustered misses: narrowed.** Clustering is not view
geometry (7 vs 1.59 after geometry stratification). Part of it is shared
between sites on the same night (co-misses 3× expected in the baseline),
which item C absorbs; the rest is site-level persistence. Independent
nights are rejected by both of us. Block shuffles (Sol: 34–93 runs) keep
real operating changes and cannot serve as the null. Agreed form: a
first-order Markov chain on Q nights after the common-mode term, gated at
the upper credible r. The anchors' r (0.37, 7/19) and the wider set's
(0.16, 12/76) differ, Fisher P = 0.055; with 19 transitions, r is set by
the gate, not estimated. Under the gate no current gap qualifies as a
departure, which is why A issues cards.

**(b) Reporting an "off" state: resolved in Sol's favour, with a
substitute.** Detectability of flames smaller than any detected flame and
the off-state detection rate are not identified; "off" cannot be read as
extinction. The identifiable statement is a bound relative to detected
flames: 4582's 14 dark nights exclude a flame like those detected at
≥ 0.2 MW (P = 2.9e-3), a drop from 14.6 MW to below about 0.2 MW. The
curve is conditioned on a second detection the same night, so it
represents flames that were detectable at least once; flames below that
class are untested.

**(c) 4582, 4583, 4497: one shared sub-field event.** 4582/4583 are one
source (Sol). 4497's darkness adds little: it had been 0.2–1.5 MW since
mid-September and its nightly lit status is not associated with the
complex (Fisher P = 0.53). The informative neighbour is 4500: 54.5, 40.5
and 59.6 MW on 10-19 to 10-21, then 0.5–0.9 MW for 11 nights from 10-22,
then 73–87 MW from 11-02; the complex's first heat since 10-20 came on
11-04 (0.2 MW). A high-flaring spell, then heat loss at two sources within
one night, then a return at 4500, fits a shared operating change (for
example a hub shut-in and restart) better than an isolated flare failure.
It does not exclude a shared flare-system failure.

## Final agreed list (my side)

1. **Thermal-departure evidence cards for bright flares.** Night-level
   score with a Markov null gated at the upper credible miss persistence,
   source-complex units, one-clear-pass eligibility, and a heat bound per
   spell. Today it produces cards, not alerts: the 4582/4583 complex
   (14.6 MW to below about 0.2 MW, 10-21 to 11-03) and 927 (6.3 MW, 6 dark
   nights, excluded from 0.3 MW up).
2. **Heat-state dating.** A censored normal / reduced / darkness model that
   dates heat changes before a dark spell is scored, so a turndown such as
   4497's (26.7 → 1.09 MW) is reported as reduced heat, not as an outage.
3. **Shared-heat-loss context.** Donor heat at 10, 50 and 100 km, the
   target's complex excluded. The 4582 card carries "shared heat loss with
   4500, 10-22 to 11-01", the leading benign alternative for SkyTruth to
   check.
4. **Small-flare darkness cards.** 3662, 3644 and 6011 (0.5–0.95 MW,
   dark 35–45 clear nights, no source within 4 km): every detected size
   class excluded, so the remaining alternatives are a flame below about
   0.2 MW or a stop.

## Remaining disagreements

- The value of r: the anchors' 0.37 (n = 19) against 0.16 (n = 76) at the
  wider set. A rolling re-estimate as nights accumulate is the test.
- Whether small-flare cards (item 4) should be surfaced to SkyTruth now;
  Sol's shortlist does not include them.
- The 4582 interpretation: I read 4500's drop within one night of the
  complex, and its later return, as a shared sub-field event; Sol's round-2 position treats 4497's
  coincidence as uninformative and does not address 4500.
