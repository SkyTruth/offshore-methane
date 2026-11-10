# Implementation brief: long-record time axes (D66)

You are implementing the design in `nightly-flaring/docs/long_record_ux.md`
(read it first, all of it). Repository root:
`/Users/jonathanraphael/.t3/worktrees/offshore-methane/t3-4e2ea75a`, branch
`t3/redesign-long-range-flaring-ux` (based on `nightly-flaring`). Work only in
`nightly-flaring/`.

## Ground rules

- Do not commit, push or publish. Do not call `build_viewer.publish_pages()`.
- Do not change pipeline outputs or the flare-out / dark-night methods.
  `flareouts.py` and `darknights.py` may only change if a test needs a
  helper; prefer doing the windowing in `build_viewer.py`.
- `outputs/` is git-ignored. Its `v5_*` segments, `gfw_oil_sites.csv`,
  basemap and MYC files are symlinks to the production run; treat them as
  read-only. Write builds only to `outputs/site/`, `outputs/site_long_1500/`,
  `outputs/site_long_5000/`.
- The main session is building the unchanged site into `outputs/site/` and
  will move it to `outputs/site_before/` (the pre-change reference for
  §4.4). Do not write `outputs/site/` until `outputs/site_before/index.html`
  exists.
- `outputs/vendor/d3.min.js` and `outputs/vendor/topojson-client.min.js` are
  local copies of the CDN scripts the page loads (d3 7.8.5, topojson-client
  3.1.0). Headless checks can route the CDN URLs to these files when there is
  no network. The page itself keeps loading from the CDN.
- Python: `/Users/jonathanraphael/.t3/worktrees/offshore-methane/t3-d5acd8c5/nightly-flaring/.venv/bin/python`.
  Playwright: `NODE_PATH=/Users/jonathanraphael/git/skinner/node_modules` with
  system Chrome (`channel: "chrome"`) if your sandbox allows; otherwise say
  so and the main session runs the headless checks.
- Style: AGENTS.md (ruff, 88 columns, `# %%` cells, param-driven `main()`, no
  argparse). Keep the viewer a single `viewer_template.html`; no build step,
  no new JS dependencies.
- A full real build reads ~1 GB of overpass CSVs. If you add a cache to speed
  iteration, put it under `outputs/` and keep the build's results identical.
- EMIT listing needs the network. If yours has none, `emit.enrich` must still
  degrade as it does today; the main session rebuilds on a networked machine.

## Scope (all of §3 of the design)

1. **Data delivery (§3.11)** in `build_viewer.py`: per-year content-hashed
   chunk files, `meta.chunks`, embedded network and regional bin totals for
   night/week/month (§3.1 rule), windowed flare-out copy with
   `record_monthly` (§3.9). Keep `encode_site_nights` (tests use it).
2. **Viewer (§3.1–§3.10, §3.12)** in `viewer_template.html`: aggregation
   helpers on 32-night block sums; bin ladder; year-aware date formatters;
   overview + focus timeline with ‹ ›, presets, Year select; step modes with
   Play; sparklines; tooltip chart; site panel record bars + calendar; regions
   Trend column; Flare-outs spell timeline and windowed charts; dark-night
   date histogram; hash state; loading and error states.
3. **Synthetic builds (§4.1)**: `dev_long_record.py` with `main(nights=...)`.
   `build_viewer.main` gains the parameters this needs (for example reusing
   flare-out / dark-night JSON from the real build, skipping EMIT), with
   defaults that keep today's behaviour.
4. **Tests**: pytest for the Python side (§4.4). A headless script (for
   example `tests/ui/check_time_axes.mjs`, not collected by pytest) that
   measures the §4.2 budgets, exercises the §4.3 parity list, runs the §4.4
   cross-checks and saves screenshots to `outputs/ui_checks/<build>/`.
5. **Docs**: D66 in `docs/decisions.md` (Choice, Alternatives, Why, Evidence,
   Status; evidence = measured budgets and check results); update "Reading
   the app" in `docs/explainer.md`, the website paragraph in `README.md`, and
   the viewer's "Method and caveats" timeline / flare-list / small-chart
   items. Record any choice the design leaves open in D66.

## Order of work

1. Data layer + pytest. 2. Viewer core: loading, block sums, range stats,
dates, timeline, step modes, playback. 3. Per-site charts and panel.
4. Regions, Flare-outs, dark nights, hash state. 5. Synthetic builds,
headless checks, budgets. 6. Docs.

## Report back

- Files changed, with one line each.
- Real build: nights, sites, `index.html` size before/after, chunk sizes.
- Synthetic builds: sizes, build time, peak memory.
- Each §4.2 budget: measured value vs budget.
- Each §4.3 parity item: pass/fail. Each §4.4 cross-check: values.
- Screenshot paths.
- Anything left undone, and every decision you made that the design did not
  fix.
