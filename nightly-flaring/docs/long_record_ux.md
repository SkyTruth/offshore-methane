# Long records: every time axis on the website (D66)

The website was built for 30 nights. The record now holds 308 nights
(2026-01-01 to 2026-11-04). NOAA's Azure copy reaches back to mid-2023
(about 1,250 more nights, D63) and NASA's L1B archive to 2012 (about 5,000
nights in all). This document lists every place the site draws time, what
breaks as nights accrue, the redesigns considered, the design chosen, and the
checks the implementation must pass. The chosen design below includes the
round-3 corrections. Decision entry: D66 in `decisions.md`.

## 1. Inventory: where time is an axis

Pixel figures assume a 1,440 px wide window (timeline strip about 1,264 px of
bars; side panel charts about 372 px).

| # | Element | Today | 30 nights | 308 | 1,500 | 5,000 |
|---|---|---|---|---|---|---|
| 1 | Timeline strip (top of Map) | one bar + one coverage cell per night; click a night, drag a range | 42 px/night | 4.1 px | 0.84 px | 0.25 px |
| 2 | Map playback | one frame per night, ≥160 ms each, ~25 s target | 25 s | 49 s | 4 min | 13 min |
| 3 | Flare-list sparkline (90×26) | whole record, nights pooled per column | 1 night/col | 4 | 18 | 57 |
| 4 | Map tooltip chart (240×62) | whole record, ≥2 px columns | 1 | 3 | 13 | 43 |
| 5 | Site panel overview (360×64) + draggable 14-night window | whole record | 1 | 2 | 9 | 29 (window is under one column) |
| 6 | Site panel pass chart | 14 local nights | fine | fine | fine | fine |
| 7 | Flare-out card charts (390 px list, 720 px detail) | whole record as evenly spaced points, plus up to 4 neighbour series | spell = 34% of width at 88 nights | 9.7% | 2% | 0.6% |
| 8 | Single-dark-night chart | ±14 nights around a run | fine | fine | fine | fine |
| 9 | Single-dark-night list | 904 runs at 88 nights; filters: EMIT, heat, region | — | ~3,200 runs | ~15,000 | — |
| 10 | Date labels, clock, header | "Mar 5"; "night 5 of 308" | no year anywhere | | | |

Failures that grow with the record:

- **Pooled columns lie.** When nights outnumber pixels, a chart column takes
  the mean heat of its nights but the *strongest* status (`ST_RANK`). A month
  with one lit night out of 25 clear nights shows the same flaring colour as a
  month lit on all 25.
- **Night picking stops working.** Below about 2 px per night, a click or an
  arrow key lands on a night the user cannot see; the site panel's 14-night
  window becomes a sub-pixel sliver.
- **Animation outgrows attention.** Nightly frames take 4 minutes at 1,500
  nights. Cloud makes adjacent frames flicker, so change over months is hard to
  see in nightly frames at any speed.
- **Flare-out evidence shrinks.** A 30-night dark spell is 4 px wide in the
  720 px chart at 5,000 nights, drawn among about 25,000 SVG points.
- **Dates lose their year.** Every label formats as "Mon D".
- **Memory and payload.** Each site's nightly heat expands to a full-length JS
  array: 7,637 × 5,000 = 38 M slots (about 300 MB). `computeRange` reads every
  night of every site on each change (38 M reads for "All" at 5,000 nights).
  The nightly strings sit inside `index.html`: measured at 308 nights, status
  strings 2.4 MB raw (0.23 MB gzip) and heat values 0.3 MB (0.10 MB gzip);
  linear growth gives about 39 MB raw at 5,000 nights. Flare-out cards embed
  their whole series (180 KB for 5 cards at 88 nights; about 10 MB at 5,000).

## 2. Options considered

### Timeline (1)

| Option | For | Against |
|---|---|---|
| A1. One strip, bins chosen by width (night/week/month) | simplest | no single-night picking on long records |
| A2. Fixed px per night, horizontal scroll | every night visible | loses the overview; 5,000 nights is 10,000 px of scrolling |
| **A3. Overview + focus** | whole record always visible; the selected range at the finest bin that fits; standard in stock and monitoring charts | two tracks take ~40 px more height |
| A4. Network calendar heatmap | shows seasonality | poor for selecting ranges; weak magnitude reading |

### Playback (2)

| Option | For | Against |
|---|---|---|
| B1. Nightly frames, faster | no new concepts | 5,000 nights at 30 fps is still 166 s; cloud flicker |
| B2. Rolling 30-night mean stepped weekly | smooth | every frame mixes 30 nights; hard to explain or read off |
| **B3. Calendar-binned frames (night, week, month), chosen by range length, user can override** | ~25 s at any length; a month frame is the same "mean over clear nights" as a range mean | one more control |
| B4. Change map (period A vs B) | best tool for "what changed" | new metric and colour scale; needs its own review (phase 2) |
| B5. Small multiples by month | no motion | needs a full-width view (phase 2) |

### Per-site charts (3–5)

| Option | For | Against |
|---|---|---|
| C1. Whole record, pooled (today) | context | pooled status misleads (above) |
| **C2. Flare-list sparkline shows the selected range**, binned to fit | the chart explains the number beside it | context lives in the tooltip and panel instead |
| C3. Context + focus inside 90 px | both | unreadable at 90 px |
| D1. Overview strip + draggable window (today) | compact | fails at 1,500+ nights |
| D2. Year rows × day-of-year columns (~1 px/night) | seasonality aligns vertically | 1 px targets |
| **D3. One calendar block per year: 53 week columns × 7 weekday rows, ~6 px cells**, plus a whole-record bar chart for magnitude | every night a clickable cell; widely known pattern (contribution graph); years stack | ~48 px per year; older years collapse |

### Flare-out evidence (7)

Window each chart to the evidence (spell ± context) on a true time scale, and
add a thin whole-record strip for context; chosen over drawing the whole record.

### Data (memory, payload)

| Option | For | Against |
|---|---|---|
| F1. Embed everything (today) | one request | 39 MB raw HTML at 5,000 nights; every rebuild invalidates all of it |
| **F2. Per-year chunk files, content-hashed, fetched in parallel at start** | only the current year changes on a nightly rebuild; small `index.html`; first paint before data | needs HTTP (already needed for pass detail) |
| F3. Monthly aggregates embedded, nightly loaded on demand | fastest at 14 years | async in every range change; month snapping; revisit past ~10 years |

## 3. Chosen design

### 3.1 One aggregation rule everywhere

Every chart bin, playback frame and range uses the range-mean rule already in
`computeRange`: per site, mean clear-sky radiant heat over nights with a
clear-sky value (clear nights with no flare count 0; a D night counts when it
has a clear-sky value), plus counts of clear nights, lit nights, observed
nights, lit-through-cloud nights and clear passes. A one-night bin reduces to
that night's value, so the per-site meaning is unchanged across units. Network totals can rise
with bin width because more sites have clear views in a longer bin.

- **Bar height** = bin mean (per site) or the sum of site bin means (network,
  region). For one night the network sum equals today's nightly bar: in the
  308-night record, 1,971 of 16,756 lit-through-cloud nights carry a
  clear-sky value, and all 1,971 are 0 MW.
- **Status strip** under per-site bars = share of clear nights lit in the bin,
  a continuous ramp from `--clear` (0%) to `--flare-5` (100%); `--cloud` when
  the bin had observations but no clear night; empty when no pass. No status
  pooling.
- **Bin units** form a ladder: night, week (ISO, Monday start), month,
  quarter, year, all calendar-aligned. A chart uses the finest unit whose bins
  are at least its minimum width (2 px for nights, 3 px otherwise, unless
  stated). Edge bins clipped by the record or range are marked partial
  ("4 of 7 nights").
- Bars, markers, shading, brushes and pointer inversion use one linear night
  scale. Partial bins span only their nights; unobserved periods stay empty.
- Every chart names its bin: "1 bar = 1 week"; list sparklines name it once
  in the list key.

### 3.2 Dates

All labels that name a day, week, month or range carry the year:
"Wed Mar 5, 2025", "Oct 6 – Nov 4, 2026", "Dec 15, 2025 – Jan 12, 2026",
"Week of Mar 3, 2025", "Mar 2025". Weekly/monthly focus ticks use `d3.utcMonth.every(k)`, with the smallest
k in 1, 2, 3, 6, 12 whose labels fit. The first kept tick in each calendar
year names the year; the bin caption has its own line. "Night 5 of 308" is dropped.

### 3.3 Timeline: overview + focus

```
Radiant heat seen each night (i)   ‹  Oct 6 – Nov 4, 2026 · 30 nights  ›   [Last 7][Last 30][Last 90][12 months][All] [Year ▾]
[▶ Play]  ┌ overview: whole record, weekly or monthly bars, brush = range ─────────────┐ 22 px
                   │ 2023        2024        2025        2026                     ▐██▌          │ 12 px year axis
                   ├ focus: the range at the finest unit that fits ─────────────────────────────┤ 56 px bars
                   │ ▁▃▅▇▅▃▁ ...                                                                │ 8 px clear-sky band
                   └ Oct 6   Oct 13   Oct 20   Oct 27   Nov 3               1 bar = 1 night ──────┘ 14 px axis
```

- **Overview** (shown when the record exceeds 92 nights): the whole record at
  weekly bins if they fit at ≥3 px, then monthly, quarterly or yearly; bar = sum of site bin means.
  A brush marks the range. Drag across bins to set a new range (snapped to
  overview bins; record edges keep night precision). Drag the brush to pan
  (length kept to the night). At All, the first drag creates a new range.
  A click within the selection in step mode keeps the range. Click a bin to set the range to that bin. Year
  labels are buttons that set the range to that calendar year.
- **Focus**: the selected range at the finest unit with nights ≥2 px or
  weeks/months ≥3 px. Bars as in §3.1; the clear-sky band below each bar is
  the mean share of all sites with clear-sky values over the bin's nights.
  The tooltip counts sites with a clear night in the bin and explains that
  wider-bin totals include sites omitted from some nightly sums.
  Hover: tooltip with the bin label, the MW sum, flaring sites, clear share,
  partial-night warning. Click a bar: step mode at that bin (§3.4). Drag:
  narrow the range to the bins dragged over.
- **‹ ›** shift a whole calendar year, quarter, month or week by that unit;
  other ranges shift by their night count, clamped to the record; keys Shift+← / Shift+→ do the same.
- **Presets**: Last 7, Last 30, Last 90 (nights), 12 months (last 365
  nights, hidden when it equals All), All; **Year** select lists the record's calendar years ("2026, to
  Nov 4" when partial). A preset shows pressed when the range matches it.
- Network bin totals for all five units are computed in Python and
  embedded as plain JSON, so the overview draws before nightly data loads.

### 3.4 Range mean, or step through it

The "Range mean | Single night" switch becomes:

```
[Mean] [Night] [Week] [Month]      Week of Mar 3, 2025 · 7 nights
```

- **Range mean**: as today.
- **Night / Week / Month** step through the range one calendar bin at a time
  (bins clipped to the range). Map, tiles, flare list, regions and legend
  show the step's values by the §3.1 rule. The Mean button is labelled
  Range mean for assistive technology. Quarter and Year appear only when the
  range spans at least two of their bins; no separate Step menu exists. Week
  and Month are disabled when the range holds fewer than two such bins.
- The list contains sites flaring in the step. Strongest uses step mean and
  night counts; Most consistent and Most clear views retain range figures,
  with that period named in each hint and the matching row subtitle.
- ← → move one step within the range; Home/End jump to the first/last step;
  Esc closes a tooltip, then the site, then the step, one layer per press. Clicking a focus bar selects that bin and its
  unit.
- **Play** animates steps in the current unit. From Range mean it picks a unit
  first: Night for ranges ≤92 nights, Week ≤730, Month beyond. Frame time =
  25 s ÷ frames, clamped to 120–1,500 ms. Values ease between adjacent steps
  as nightly frames do today; reduced motion steps without easing. Pause
  leaves the step selected. Playback updates open-site markers in place
  and does not write the URL until Pause.
- The map clock names the step: "Wed Mar 5, 2025 · passes near 01:30 local",
  "Week of Mar 3, 2025 · 7 nights · mean per site", "Mar 2025 · 31 nights ·
  mean per site", or "Mean Oct 6 – Nov 4, 2026 · 30 nights".
- Tile labels follow: "sum this night", "sum of site means this week",
  "Clear this month: sites with a clear night". A clear night for these site
  counts has status F or C, matching the old single-night count; heat means
  retain finite D values. The tile and bin tooltip use the same definition.

### 3.5 Flare-list sparklines

The 90×26 chart shows the selected range (not the whole record), binned to
fit, bars scaled to the site's highest bar in that span, strip per §3.1, and a
marker at the current step. The tooltip and the site panel carry the whole
record.

### 3.6 Map tooltip chart

The whole record, binned to fit 240 px (≥2 px per bin), strip per §3.1, the
range shaded, the current step marked, first and last dates with years and
year boundary ticks, and the bin named ("1 bar = 1 month").

### 3.7 Site panel

```
Site 4582 · Persian Gulf · QA                                         [Close]
MEAN HEAT  GAS FLARED  FLARING   CLEAR PASSES        (range figures, unchanged)
Record · Jun 2023 – Nov 2026 · 1 bar = 1 month                  max 31 MW
▂▃▅▅▆▇▇▆▅▃▂▁ ... ▅▆▇█▇▆   (range shaded; click a bar: passes for that bin)
2026 ▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦         J F M A M J J A S O N D
2025 ▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦     (53 × 7 cells per year)
[Show 2 earlier years]
Passes Mar 3 – 16, 2025 · 14 nights                                    ‹  ›
(pass chart as today)
```

- **Record bars**: whole record by the ladder (≥3 px); bin mean; strip per
  §3.1; range shaded; step marked. Click a bar: move the pass window there.
- **Calendar**: one block per year, newest first; 53 week columns × 7
  weekday rows; cell size from the panel width (about 6 px). Cell colour:
  F by the map's heat ramp, D `--flare-2`, C `--clear`, K `--cloud`,
  N empty. Nights outside the range dimmed; pass-window nights outlined in
  `--focus`; the current step outlined in `--accent`. Hover names the night
  and its value; click centres the pass window on it. More than four years:
  the four newest and the years containing the range end, step and pass window
  show; a button reveals the rest. The grid has one roving
  tab stop: arrows move by day/week, PageUp/PageDown by month, Enter/Space
  select. Focus survives updates and consumed cell keys never reach the map. Native
  buttons retain Enter/Space; Escape reaches the page. Grid cells have row
  parents and the cursor starts on the last pass-window night.
- **Pass window**: 14 nights as today; in step mode it follows the step (the
  Night keeps 14 centred nights; longer units show the step itself when
  ≤21 nights, else its last 14 nights). Outline every step night in the
  window. Labels carry years.

### 3.8 Regions table

A "Trend" column: a 64×18 sparkline of the region's sum of site means over
the whole record (ladder, ≥2 px), range shaded. It covers all sites in the
region, independent of the list filters (stated in the column's (i)).
Embedded from Python with the network totals.

### 3.9 Flare-outs tab

- **Spell timeline** above the card list: x = earliest spell start minus 60
  nights through as-of; the caption names the full record start. Ticks use
  months under two years and years beyond;
  one lane per card (tier colour, at most 14 px apart), a bar from spell start
  to end, an open
  right end with an arrow for ongoing spells; click selects the card.
- **Card charts** on a UTC time scale (gaps proportional to time). Detail
  chart: [start − 60 nights, (end, or as-of while ongoing) + 14 nights],
  clipped to the record; list chart: [start − 30, end + 7]. Above the detail
  chart, a 24 px whole-record strip of the source's monthly mean heat with the
  window marked. Dates carry years.
- **Payload**: `build_viewer` embeds a windowed copy (card and neighbour
  series cut to [start − 90, end + 30]) plus `record_monthly` per card,
  `[[YYYY-MM, mean MW, lit nights, clear nights], …]`, computed from the full
  series. `flareouts.json` on disk keeps the full series.

### 3.10 Single dark nights

- A histogram of dark runs per month across the record (per week when the
  record is under a year) with a brush that filters the list by run date;
  "All dates" resets it; the count line names the date filter. Labelled From/To
  date inputs give keyboard access; a count scale and per-bar hover name runs.
- Chart and row dates carry years.

### 3.11 Data delivery

- `index.html` keeps metadata, per-site static fields, the night list,
  plain JSON network totals (all five units) and regional totals
  (week, month, quarter, year) and the flare-out
  copy. Nightly per-site data moves to `data/nights-<YYYY>.<hash8>.json`,
  one per calendar year: statuses and sparse heat for F/D nights that year.
  Clear-pass digits for all sites have separate year-local files,
  `data/clearpasses-YYYY.<hash8>.json`, listed by `meta.clearpasses` and
  fetched after the first usable map render. Until they arrive, Most clear
  views is disabled with a loading label and Clear passes shows “…”; all
  other controls remain usable. All-unobserved site-years are
  omitted and restored as empty. A year's bytes depend only on its own nights. `meta.chunks` lists them; an unchanged year
  keeps its file name across builds.
- The page draws the header, overview and basemap first, shows
  "Loading nights…", fetches all chunks in parallel, then renders. A failed
  fetch names the file and HTTP status and offers Retry. A 404 triggers one
  guarded reload; successful loading clears the guard. Retry after a 404
  refreshes the index without cache and uses its current file names; HTTP advice appears only for `file:`. Publication retains
  the preceding index's referenced night and clear-pass files for one build. While loading,
  summaries have placeholders and time controls are disabled.
- In memory: status strings per site; heat as one `Float64Array(NN)` (NaN
  when absent) per ever-lit site; clear passes as one `Uint8Array(NN)`
  whenever a chunk supplies digits. Per-site 32-night block sums (sum, finite-value nights,
  lit, observed, lit-through-cloud, passes, F/C nights) make any range or bin
  O(blocks + 64) per site.
- `passes/<id>.json` is unchanged.

### 3.12 Shareable state

`#map?from=YYYY-MM-DD&to=YYYY-MM-DD&step=night|week|month&at=YYYY-MM-DD&site=<id>`
restores range, step and opened site; the page updates it with
`history.replaceState` (no history spam). Incoming hashes restore before any serialization; omitted
site closes detail, and dates clamp to the record. Old `#map` links still work.

### 3.13 Unchanged

Map encoding (colour, size, rings), the radiant-heat filter, flare-list
orders, EOG comparison card, How it works text other than "Reading the app",
explainer figures (each caption names its fixed period), flare-out and
dark-night methods.

## 4. Checks

### 4.1 Synthetic long records (development only)

`dev_long_record.py`, `main(nights=1500)` / `main(nights=5000)`, builds a
site in `outputs/site_long_<n>/` from the real segments shifted back by whole
calendar years (2026 stays real, so Nov 5 – Dec 31 of earlier years is
unobserved, which tests gaps). Per (site, year): heat × a seeded factor in
0.6–1.4, and with probability 0.15 the site is dark that year (F → C), so
trends and start/stop show. Pass files and flare-out / dark-night data come
from the real period only. `meta.synthetic = true` puts a "Synthetic test
data" banner on the page. Peak memory ≤ 8 GB.

### 4.2 Budgets (5,000-night synthetic build, desktop Chrome, local server)

| Measure | Budget |
|---|---|
| `index.html` growth per 1,000 nights | ≤ 150 KB |
| First paint (header + overview) | ≤ 1 s |
| Interactive (map drawn after chunks load) | ≤ 4 s |
| Range statistics, "All" | ≤ 30 ms |
| Range change → full refresh, with/without site and all years | ≤ 150 ms |
| Playback, monthly frames, with/without site and all years | median ≤ 42 ms, p95 ≤ 67 ms |
| JS heap after load | ≤ 350 MB |

The 1,500-night build: interactive ≤ 2 s.

### 4.3 Parity: interactions that must still work

Click a night; drag a range; presets; ← → Home End Esc; Play/Pause; radiant-heat
filter keeps the same sites while stepping and playing; Flaring only; list
orders and "Show more"; hover a list row highlights the map; open a site from
map or list; pass chart window ‹ › and loading; regions zoom; Flare-outs card
selection, Show on map, card hash links; dark-night filters, neighbour links,
hash links; How it works rail; phone layout at 390 px.

### 4.4 Correctness

- Python: chunk round trip equals `encode_site_nights` over the whole record;
  year boundaries; hashed names stable for unchanged content; network and
  regional bin totals equal a direct pandas computation of the §3.1 rule;
  windowed flare-out series and `record_monthly`.
- Headless: for a month bin, the overview tooltip MW equals the Radiant heat
  tile after selecting that month as the range, and equals the Month step's
  tile; Range mean over "All" equals the pre-change build on the real 308-night
  record (tiles, top-25 list, region table) to the displayed precision.
- Screenshots: 1,440 / 1,024 / 390 px × the 308-night real build and both
  synthetic builds, for Map (range, night, week, month, site open), Flare-outs
  and Single dark nights.

## 5. Deferred

- **Change map (B4)** and **small multiples (B5)**: the strongest tools for
  multi-year change; they need a change metric with its own noise model and
  review.
- **Validation by year**: the EOG card compares any range with MYC 2025; with
  2023–2025 nights it could compare each year with its own MYC year.
- **Flare-out method on long records**: `flareouts.screen` calibrates on the
  record's first 29 nights and freezes source centres there
  (`baseline_end`); on a multi-year record that anchors calibration to the
  oldest month. Needs a method change and review before long backfills feed
  the Flare-outs tab.
- **Progressive loading** per year (F3) once the record passes ~10 years.
