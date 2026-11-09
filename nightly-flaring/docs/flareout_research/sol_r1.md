# Sol — Round 1: expected thermal activity, then unexpected absence

These models identify departures from a site's observed operating history. They cannot establish that production continued, that a flare *should* have remained lit, or that methane escaped.

I used only the two v5 segments, GFW site coordinates, and nightly-flaring methods. There are 672,056 site-nights, 2,417,680 deduplicated nighttime passes, and 651/7,637 sites with a strict-clear detection (736 with any detection). Never-detected infrastructure is unclassifiable from this record alone; the 2025 catalogue did not define expectations.

Baseline: **August 10–September 7**; evaluation: **September 8–November 4, 58 nights**. August 9 is excluded from fitting. A qualified night, Q, has ≥2 deduplicated SZA≥97° passes with `cloud_mask` 0/1. Cloudy/missing passes supply no negative evidence; **any detection, including through cloud, interrupts a complete-blackout run**. Using 0/1 for denominators avoids the outcome-dependent isolated-cloud override in `strict_clear`.

## 1. Opportunity-time departure from a persistent baseline

1. **Target/null.** Detect unexpectedly long absences at previously bright, frequently observed flares. Null: the baseline detection process continues.
2. **Model.** For N baseline Q nights and S nights with any detection, use q∼Beta(S+1,N−S+1). After k qualified misses, T(k)=B(S+1,N−S+1+k)/B(S+1,N−S+1). Score once per night, not independently per satellite. A fuller emission model is logit(pᵢₚ)=αᵢ+γ_sat+f(|col−1599.5|)+β·`cloud_clear_neighbors`+g(log Hᵢₜ), with pooled coefficients and latent heat H; the feasibility calculation uses constant nightly q.
3. **Computation.** Join segments, run `dedupe_overpasses`, reconstruct local `night` from `time_utc` and `lon`; use `detected`, `cloud_mask`, `sza`, `rh_mw`. Eligibility: N≥12, S/N≥0.8, median positive clear-pass heat ≥3 MW.
4. **Controls.** Eligibility excludes intermittent/weak baselines; exposure counts handle missing satellites. Night-level scoring avoids multiplying correlated same-night misses. Residual cloud errors, changed viewing geometry and serial dependence remain in the constant-q shortcut.
5. **Feasibility.** **21 anchors**, including 16 Persian Gulf sites; none in the Gulf of Mexico. With M=21×58=1,218 possible starts, flag T≤0.05/M: **five gaps at four sites** (table). Under this stipulated posterior-predictive null, expected false alerts ≤0.05; 20,000 simulations produced **382/20,000=0.0191 alerts/run**, not an estimate of real-world false accusations.
6. **Triage/validation.** Inspect multisatellite pass histories and apply ideas 2–4. Ideal full-evidence erasures on the actual observation schedule detected **863/1,092 seven-night windows** and **884/945 fourteen-night windows**; these measure scheduling power, not causal specificity. Validate forward and with temporal blocks.
7. **Cannot tell.** A changed operating regime, benign shutdown, or shrinking flare can reject this null without any flare-system failure.

| Site | Initial site-level gap, 2026 | Qualified misses k | M×T | Prior-week source heat: median MW (clear source groups) | Nights with source within 1 km during gap |
|---|---|---:|---:|---:|---:|
| 4582 | Oct 21–Nov 4 | 15 | 0.000010 | 12.13 (25) | 1/15 |
| 4497 | Oct 23–30 | 8 | 0.000052 | 0.56 (14) | 3/8 |
| 3669 | Oct 23–Nov 4 | 10 | 0.00258 | 0.36 (9) | 0/13 |
| 4286 | Oct 3–6 | 4 | 0.0387 | 5.51 (14) | 4/4 |
| 4497 | Oct 15–18 | 4 | 0.0387 | 0.45 (11) | 4/4 |

Source heat uses copy-peak cloud mask 0/1, not cloud-dimmed detections. All significance numbers are nominal under the simplified null; these exploratory thresholds require independent calibration.

## 2. Source-centred tracking instead of platform-ID absence

1. **Target/null.** Detect disappearance of a thermal source while allowing its assigned platform ID to change. Null: the same source remains observable somewhere inside its positional uncertainty.
2. **Model.** Model copy-group centroids zⱼ∼Σₖπₖ Normal(μₖ,Σ_scan); platform assignment is a nuisance. Test source detections Yₖₜ=1{a centroid falls within rₖ of μₖ} using idea 1. Initial proxy: baseline median centroid, fixed r=1 km.
3. **Computation.** Concatenate `copies`; deduplicate (`satellite`,`granule`,`group_id`). Use `centroid_lat/lon`, `assigned_site`, `group_rh_mw`, `cloud_mask_peak`, `peak_col`, `time_utc`. Keep unassigned and cloudy detections as positive evidence.
4. **Controls.** Candidate baseline centroid r95 values are 531–743 m. Check 0.75/1/1.5/2-km sensitivity. Avoid indiscriminate connected-component totals: the 2-km site graph has **4,609 components**, the largest containing **97 sites**; unrelated flares could conceal a loss.
5. **Feasibility.** A 1-km aperture interrupts both four-night alerts entirely. Re-counting source-aperture runs leaves **three nominal gaps**: 4582 (14 qualified nights), 3669 (10), 4497 (5). Site 4497's apparent Oct 15–18 blackout has five source groups assigned to **4485**, whose platform is **2.51 km** away; a 2-km platform-neighbour veto misses this. Used solely as a veto, this cannot increase idea 1's false-alert count; a refitted source model needs recalibration.
6. **Triage/validation.** Show centroids, source heat and assignment histories; measure held-out-satellite localization and robustness to synthetic assignment swaps. Ambiguous overlapping tracks trigger abstention.
7. **Cannot tell.** A nearby source need not be the original flare. Site-pixel clouds do not directly measure cloud at the learned centroid; negative evidence remains approximate.

## 3. Local exceptions to shared shutdowns

1. **Target/null.** Find site-specific thermal losses while comparable nearby infrastructure remains active. Null: a shared field/regional state explains the reduction.
2. **Model.** logit(qᵢₜ)=αᵢ+βxᵢₜ+u_region,t+v_field,t+δᵢ1(t≥τ). Compare δᵢ=0 against δᵢ<0, estimating shared effects without the target's source/assignment component.
3. **Computation.** Use Q/detection histories, site coordinates, `regions.region_of`, and historical co-activity. Collapse nearby donor anchors into component-night units; compare several geographic scales.
4. **Controls.** Common darkness lowers suspicion rather than creating many independent alarms. Match exposure/scan composition; keep clouds as unknown. Correlated operational changes need block calibration; geography does not establish shared gas supply.
5. **Feasibility.** As a descriptive gate, donor activity ≥90% retains **all five** initial gaps: **90.7–97.7%** lit across 11 other Persian Gulf components. Within 100 km, 4582's donors were lit **57/58 component-nights** (four components); 3669's were **28/29** (three). This veto inherits the first screen's nominal false-alert bound; the hierarchical model is not fitted. No Gulf of Mexico anchor exists to calibrate a hurricane-season null here.
6. **Triage/validation.** Rank local exceptions above synchronous losses. Hold out whole fields and time blocks; inject synchronous and single-source outages separately and measure confusion.
7. **Cannot tell.** A platform-specific maintenance stop or gas-recovery project also looks like a local exception. Active neighbours do not prove continuing production at the target.

## 4. Separate major heat loss from final disappearance below sensitivity

1. **Target/null.** Distinguish bright→dark events from an already diminished flare becoming undetectable. Null: a small, still-burning flare continues below variable sensitivity.
2. **Model.** A semi-Markov bright/dim/dark state drives log Hₜ=μ_state+φ(log Hₜ₋₁−μ_state)+εₜ. Use idea 1's detection likelihood and log RHₚ∼Normal(log Hₜ+view_col+sat_bias,σ²) for detections; integrate undetected heat rather than assigning physical zero. Compare gradual dimming and abrupt-loss models.
3. **Computation.** Use pass `rh_mw`, `temp_k`, `fit_flag`, `col`, satellite and source coordinates. Estimate detection/geometry terms from same-night satellite discordance, with partial pooling.
4. **Controls.** The sensitivity floor is not a hard 1 MW cutoff: **1,855/24,006 clear-flag detections are below 0.3 MW**. Detectability below threshold is unidentifiable from positive detections alone. Test sensitivity curves and state-duration priors explicitly.
5. **Feasibility.** A proxy gate requiring ≥3 prior-week clear source groups and median ≥3 MW leaves **one of the three source-gap candidates: 4582**. Its Oct 21–Nov 3 interval has **45 clear passes**, 44 with ≥6 clear neighbours, across all three satellites; no recorded source lies within **3 km**. A 0.19-MW group appears Nov 4, 626 m from the baseline centre. For 3669, the earlier heat change is Aug 29: medians **5.98→0.32 MW** in adjacent fourteen-night windows (45/16 source groups). Dating its event Oct 23 loses that history.
6. **Triage/validation.** Separate “major heat loss with residual flame” from “no thermal evidence.” Inject sustained small flames and gradual declines, alongside complete erasures; measure false extinction labels and onset error. A veto inherits the first screen's nominal bound; state-model error is unmeasured.
7. **Cannot tell.** Even 4582 remains an inspection priority, not a confirmed unlit/venting event. Heat alone cannot distinguish zero flame from sufficiently small flame, gas diversion, or halted production.

## Questions for my partner

- Can source-centred eligibility recover coverage beyond these 21 anchors without merging distinct flares?
- Would you treat a weak residual source as a veto or a separate partial-system-loss hypothesis? How would you validate that distinction internally?
- What serial-dependence null would you trust with only 29 baseline nights? Can it invalidate these five nominal alerts?
- Are the local donors credible controls for 4582, or could a shared operating change escape both geographic screens?

Reproduction and numeric outputs: [`scripts_sol/README.md`](scripts_sol/README.md). No external data, annual catalogue matching, downloads or partner memos were used. The centroid-mixture, hierarchical and semi-Markov fits are proposals; the aperture, donor, brightness and run screens above were executed.
