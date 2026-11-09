"""D62 selection, EMIT timing, bounded imagery and build refresh, all offline."""

import json
from datetime import date, timedelta
from unittest.mock import Mock

import build_viewer as b
import darknights as d
import emit
import flareouts as f
import numpy as np
import pandas as pd
import pytest
import update
from test_flareouts import history_fixture


@pytest.fixture(autouse=True)
def no_live_services(monkeypatch):
    """A missed mock must never reach CMR or Earth Engine in these tests."""
    monkeypatch.setattr(
        d.requests, "get", Mock(side_effect=AssertionError("offline test"))
    )
    monkeypatch.setattr(
        emit, "_ee_client", Mock(side_effect=AssertionError("offline test"))
    )


def frames(days=45, dark=(20,), cloudy=(), cloudy_lit=(), missing_heat=()):
    """One source with two real passes/night and controlled source evidence."""
    nights, passes, copies = [], [], []
    for i, dt in enumerate(pd.date_range("2026-08-01", periods=days)):
        night = dt.strftime("%Y-%m-%d")
        nights.append({"site_id": 1, "night": night, "lat": 0.0, "lon": 0.0})
        for j, sat in enumerate(("SNPP", "NOAA-20")):
            t = (dt + pd.Timedelta(days=1, hours=j)).isoformat()
            granule = f"g{i}_{j}"
            lit = i not in dark or i in cloudy_lit
            passes.append(
                {
                    "site_id": 1,
                    "satellite": sat,
                    "granule": granule,
                    "time_utc": t,
                    "sza": 140,
                    "cloud_mask": 3 if i in cloudy else 0,
                    "detected": lit,
                    "rh_mw": 2 if lit else 0,
                    "dist_m": 10,
                    "col": 1100 if j == 0 else 2600,
                }
            )
            if lit:
                copies.append(
                    {
                        "satellite": sat,
                        "granule": granule,
                        "group_id": 1,
                        "time_utc": t,
                        "assigned_site": 1,
                        "centroid_lat": 0.0,
                        "centroid_lon": 0.0,
                        "cloud_mask_peak": 3 if i in cloudy_lit else 0,
                        "group_rh_mw": np.nan if i in missing_heat else 2.0,
                    }
                )
    return tuple(pd.DataFrame(rows) for rows in (nights, passes, copies))


def screen(*args, **kwargs):
    return d.screen(*args, cards=kwargs.get("cards", {"candidates": []}))


def night_rows(payload):
    """Expand compact fields only for assertions about observable night evidence."""
    rows = []
    for run in payload["runs"]:
        source = payload["sources"][run["source_id"]]
        for night in run["nights"]:
            expected = night["expected_misses"]
            curve = (
                payload["detection_curve"][expected["curve"]]
                if expected["curve"] is not None
                else {}
            )
            burning = night["burning_rate"]
            rows.append(
                {
                    **night,
                    "id": source["unit"]["id"] + "_" + night["night"],
                    "unit": source["unit"],
                    "run": {
                        "start": run["start"],
                        "end": run["end"],
                        "clear_dark_nights": run["clear_dark_nights"],
                    },
                    "card_id": run["card_id"],
                    "bracket": run["bracket"],
                    "burning_rate": {
                        **burning,
                        "rate": burning["lit"] / burning["qualified"],
                        "window_start": str(
                            (
                                pd.Timestamp(night["night"]) - pd.Timedelta(days=14)
                            ).date()
                        ),
                        "window_end": str(
                            (
                                pd.Timestamp(night["night"]) + pd.Timedelta(days=14)
                            ).date()
                        ),
                    },
                    "expected_misses": {
                        "p": curve.get("probability"),
                        "n": curve.get("trials", 0),
                        "detections": curve.get("detections", 0),
                        "k": expected["k"],
                        "miss_all": expected["miss_all"],
                        "lower_mw": curve.get("lower_mw"),
                        "upper_mw": curve.get("upper_mw"),
                    },
                    "passes": [
                        {
                            "satellite": payload["pass_satellites"][sat],
                            "time_utc": d._stamp(
                                pd.Timestamp(night["night"], tz="UTC")
                                + pd.Timedelta(seconds=seconds)
                            ),
                            "clear": clear,
                            "view_zone": payload["pass_view_zones"][zone],
                            "detected": detected,
                        }
                        for sat, seconds, clear, zone, detected in night["passes"]
                    ],
                    "emit": {
                        "status": source["emit_status"],
                        "scenes": [
                            {
                                **source["scenes"][sid],
                                "phase": phase,
                                "hours_from_dark_pass": hours,
                            }
                            for sid, phase, hours in run["emit_scenes"]
                        ],
                    },
                }
            )
    return rows


def controls(payload, uid="1"):
    source = payload["sources"][uid]
    return [
        {**source["scenes"][sid], "phase": "lit on both sides"}
        for sid in source["comparison_scenes"]
    ]


def at(payload, day):
    night = (date(2026, 8, 1) + timedelta(days=day)).isoformat()
    return next(n for n in night_rows(payload) if n["night"] == night)


def entries(*times):
    return [
        {
            "producer_granule_id": "EMIT_L1B_RAD_002_"
            + t.replace("-", "").replace(":", "").replace("Z", ""),
            "time_start": t,
            "cloud_cover": "12",
        }
        for t in times
    ]


def listing(monkeypatch, *times):
    cmr = Mock(return_value=entries(*times))
    monkeypatch.setattr(emit, "cmr_granules", cmr)
    return cmr


def test_other_dark_run_nights_count_in_window_and_recent_heat_counts_nights():
    payload = screen(*frames(dark=range(18, 23)))
    row = at(payload, 20)
    assert row["run"]["clear_dark_nights"] == 5
    assert row["burning_rate"] == {
        "rate": 24 / 28,
        "lit": 24,
        "qualified": 28,
        "window_start": "2026-08-07",
        "window_end": "2026-09-04",
    }
    assert row["recent_heat_mw"] == 2.0
    assert row["recent_heat_n"] == 24  # nightly medians, not 48 groups
    assert row["expected_misses"]["k"] == 2
    assert row["passes"][0]["view_zone"] == "near nadir"
    assert row["passes"][1]["view_zone"] == "swath edge"
    assert all(not p["detected"] for p in row["passes"])


def test_strict_more_than_half_and_at_least_ten_qualified(monkeypatch):
    n, p, c = frames()
    original = f.unit_history

    def limited(*args):
        h, cp, obs = original(*args)
        # Ten other qualified nights, five lit and five isolated clear misses.
        h["qualified"] = False
        for i in [8, 10, 12, 14, 16, 24, 26, 28, 30, 32, 20]:
            h.iloc[i, h.columns.get_loc("qualified")] = True
        for i in [8, 10, 12, 14, 16]:
            h.iloc[i, h.columns.get_loc("lit")] = False
        return h, cp, obs

    monkeypatch.setattr(f, "unit_history", limited)
    assert not any(r["night"] == "2026-08-21" for r in night_rows(screen(n, p, c)))

    def above(*args):
        h, cp, obs = limited(*args)
        h.iloc[8, h.columns.get_loc("lit")] = True
        return h, cp, obs

    monkeypatch.setattr(f, "unit_history", above)
    assert at(screen(n, p, c), 20)["burning_rate"]["qualified"] == 10

    def too_few(*args):
        h, cp, obs = above(*args)
        h.iloc[10, h.columns.get_loc("qualified")] = False
        return h, cp, obs

    monkeypatch.setattr(f, "unit_history", too_few)
    assert not any(r["night"] == "2026-08-21" for r in night_rows(screen(n, p, c)))


def test_cloud_only_nights_do_not_break_run_but_cloudy_detection_does():
    payload = screen(*frames(dark=(19, 20, 21), cloudy=(20,)))
    row = at(payload, 19)
    assert [n["night"] for n in payload["runs"][0]["nights"]] == [
        "2026-08-20",
        "2026-08-22",
    ]
    assert row["run"] == at(payload, 21)["run"]
    assert not any(r["night"] == "2026-08-21" for r in night_rows(payload))
    payload = screen(*frames(dark=(19, 20, 21), cloudy=(20,), cloudy_lit=(20,)))
    assert at(payload, 19)["run"]["clear_dark_nights"] == 1
    assert at(payload, 21)["run"]["clear_dark_nights"] == 1


def test_cloudy_source_copy_with_wrong_assignment_still_makes_night_lit():
    n, p, c = frames(dark=(19, 20, 21), cloudy_lit=(20,))
    c.loc[c.granule.str.startswith("g20_"), "assigned_site"] = 900
    p.loc[p.granule.str.startswith("g20_"), "detected"] = False
    payload = screen(n, p, c)
    assert not any(r["night"] == "2026-08-21" for r in night_rows(payload))
    assert at(payload, 19)["bracket"]["recovery"] == "2026-08-22T00:00:00Z"


def test_provisional_boundary_and_unknown_heat_are_retained():
    payload = screen(*frames(days=45, dark=(30, 31, 44), missing_heat=range(45)))
    assert not at(payload, 30)["provisional"]  # exactly 14 later nights
    assert at(payload, 31)["provisional"]
    row = at(payload, 44)
    assert row["provisional"]
    assert row["recent_heat_mw"] is None and row["recent_heat_n"] == 0
    assert row["expected_misses"]["p"] is None
    assert row["expected_misses"]["miss_all"] is None
    assert row["bracket"]["recovery"] is None


def test_recent_heat_is_median_of_lit_clear_nights_only():
    n, p, c = frames(dark=(20,), cloudy=(10,), cloudy_lit=(10, 12), missing_heat=(11,))
    c.loc[c.granule.str.startswith("g13_"), "group_rh_mw"] = 1000
    row = at(screen(n, p, c), 20)
    assert row["recent_heat_mw"] == 2.0
    assert row["recent_heat_n"] == 25  # cloud-only, cloudy heat, NaN excluded


def test_known_detection_curve_and_sort(monkeypatch):
    curve = [
        {
            "lower_mw": 1,
            "upper_mw": 2,
            "trials": 40,
            "detections": 30,
            "probability": 0.75,
        }
    ]
    monkeypatch.setattr(f, "detection_curve", lambda p: curve)
    payload = screen(*frames(dark=(20, 40)))
    row = at(payload, 20)
    assert row["expected_misses"] == {
        "p": 0.75,
        "n": 40,
        "detections": 30,
        "k": 2,
        "miss_all": 0.0625,
        "lower_mw": 1,
        "upper_mw": 2,
    }
    assert night_rows(payload) == sorted(night_rows(payload), key=d.default_key)
    assert d._expected(2.01, 2, curve)["p"] is None


@pytest.mark.parametrize(
    "dark,scene,phase,last_dark,recovery",
    [
        (
            (20,),
            "2026-08-21T15:00:00Z",
            "boundary",
            "2026-08-22T01:00:00Z",
            "2026-08-23T00:00:00Z",
        ),
        (
            (20, 21, 22),
            "2026-08-23T15:00:00Z",
            "during",
            "2026-08-24T01:00:00Z",
            "2026-08-25T00:00:00Z",
        ),
    ],
)
def test_brackets_and_phase_labels(
    monkeypatch, tmp_path, dark, scene, phase, last_dark, recovery
):
    payload = screen(*frames(dark=dark))
    listing(monkeypatch, scene)
    result = d.enrich(payload, imagery=False, cache_dir=tmp_path)
    row = at(result, 20)
    assert row["bracket"] == {
        "last_lit": "2026-08-21T01:00:00Z",
        "first_dark": "2026-08-22T00:00:00Z",
        "last_dark": last_dark,
        "recovery": recovery,
    }
    assert row["emit"]["scenes"][0]["phase"] == phase
    assert row["emit"]["scenes"][0]["hours_from_dark_pass"] == -9


def test_calendar_month_listing_per_source_nearby_window_and_controls(
    monkeypatch, tmp_path
):
    payload = screen(*frames(dark=(20, 21, 22)))
    cmr = listing(
        monkeypatch,
        "2026-08-17T15:00:00Z",
        "2026-08-19T15:00:00Z",
        "2026-08-20T15:00:00Z",
        "2026-08-21T15:00:00Z",
        "2026-08-25T15:00:00Z",
        "2026-08-26T15:00:00Z",
    )
    monkeypatch.setattr(
        emit,
        "cached_chips",
        lambda *a: {"status": "available", "source_coverage": "covered"},
    )
    result = d.enrich(payload, cache_dir=tmp_path)
    assert cmr.call_count == 2
    assert {c.args[0] for c in cmr.call_args_list} == set(emit.PRODUCTS)
    assert {c.args[3:5] for c in cmr.call_args_list} == {
        ("2026-08-01", "2026-09-30"),
    }
    scenes = at(result, 20)["emit"]["scenes"]
    assert len(scenes) == 4  # Aug 19 noon through Aug 26 noon
    assert [s["phase"] for s in scenes] == ["before", "before", "boundary", "after"]
    comparison = controls(result)
    assert len(comparison) == 2
    assert all(s["phase"] == "lit on both sides" for s in comparison)
    assert all(s["time_utc"] != "2026-08-21T15:00:00Z" for s in comparison)
    assert comparison[0]["time_utc"] == "2026-08-26T15:00:00Z"


def test_controls_require_both_nearest_passes_not_just_a_lit_night():
    passes = [
        {"time_utc": f"2026-08-20T{h:02}:00:00Z", "detected": lit}
        for h, lit in [(0, True), (1, False), (20, True), (23, True)]
    ]
    dark = [d._instant("2026-08-21T00:00:00Z")]
    scenes = [
        {"scene_id": str(h), "time_utc": f"2026-08-20T{h:02}:00:00Z"}
        for h in (0, 15, 21, 23)
    ]
    assert [s["scene_id"] for s in d.comparison_scenes(scenes, passes, dark, [])] == [
        "21"
    ]


def test_render_bound_deduplicates_prioritizes_nearby_and_copies_cache(
    monkeypatch, tmp_path
):
    payload = screen(*frames(dark=(20, 21, 22)))
    listing(
        monkeypatch,
        "2026-08-21T15:00:00Z",
        "2026-08-22T15:00:00Z",
        "2026-08-23T15:00:00Z",
        "2026-08-10T15:00:00Z",
    )
    monkeypatch.setattr(emit, "_ee_client", lambda: Mock())
    monkeypatch.setattr(emit, "cached_chips", lambda *a: None)
    folder = tmp_path / "chips"
    folder.mkdir()
    for name in ("ch4.png", "rgb.png"):
        (folder / name).write_bytes(b"cached png")
    render = Mock(
        return_value={
            "status": "available",
            "source_coverage": "covered",
            "cache_folder": str(folder),
        }
    )
    monkeypatch.setattr(emit, "fetch_chips", render)
    result = d.enrich(
        payload, site_dir=tmp_path / "site", cache_dir=tmp_path, max_renders=2
    )
    assert render.call_count == 2
    assert render.call_args_list[0].args[0]["time_utc"] == "2026-08-21T15:00:00Z"
    for night in night_rows(result):
        assert [s["imagery"]["status"] for s in night["emit"]["scenes"]] == [
            "available",
            "available",
            "queued",
        ]
        assert "later build" in night["emit"]["scenes"][-1]["imagery"]["message"]
    assert controls(result) == []  # unmeasured coverage cannot be a comparison
    assert len(list((tmp_path / "site" / "img" / "emit").glob("*.png"))) == 4
    assert result["rendering"]["new_renders"] == 2
    assert result["counts"]["with_chips"] == 3


def test_cached_chips_survive_offline_and_do_not_spend_bound(monkeypatch, tmp_path):
    listing(monkeypatch, "2026-08-21T15:00:00Z", "2026-08-22T15:00:00Z")
    folder = tmp_path / "cached"
    folder.mkdir()
    for name in ("ch4.png", "rgb.png"):
        (folder / name).write_bytes(b"cached png")
    monkeypatch.setattr(
        emit,
        "cached_chips",
        lambda *a: {
            "status": "available",
            "source_coverage": "covered",
            "cache_folder": str(folder),
        },
    )
    client = Mock(side_effect=AssertionError("must use cache first"))
    monkeypatch.setattr(emit, "_ee_client", client)
    result = d.enrich(
        screen(*frames()), site_dir=tmp_path / "site", cache_dir=tmp_path, max_renders=0
    )
    assert not client.called
    assert result["rendering"]["new_renders"] == 0
    assert result["counts"]["with_chips"] == 1


def test_offline_listing_degrades_without_error_or_secrets(tmp_path):
    session = Mock()
    session.get.side_effect = OSError("private token")
    result = d.enrich(screen(*frames()), cache_dir=tmp_path, session=session)
    assert at(result, 20)["emit"]["status"] == "unavailable"
    assert at(result, 20)["emit"]["scenes"] == []
    assert session.get.call_count == 2
    assert "private token" not in json.dumps(result)


def test_cached_and_partial_listing_keep_scenes(monkeypatch, tmp_path):
    cached = entries("2026-08-21T15:00:00Z")
    cmr = Mock(side_effect=[emit.CachedListing(cached), [], [], []])
    monkeypatch.setattr(emit, "cmr_granules", cmr)
    result = d.enrich(screen(*frames()), cache_dir=tmp_path, imagery=False)
    assert at(result, 20)["emit"]["status"] == "partial"
    assert len(at(result, 20)["emit"]["scenes"]) == 1


def test_ee_connection_failure_is_graceful_and_retried(monkeypatch, tmp_path):
    listing(monkeypatch, "2026-08-21T15:00:00Z")
    client = Mock(side_effect=OSError("private token"))
    monkeypatch.setattr(emit, "_ee_client", client)
    for _ in range(2):
        result = d.enrich(screen(*frames()), cache_dir=tmp_path)
        assert at(result, 20)["emit"]["scenes"][0]["imagery"]["status"] == "unavailable"
        assert "private token" not in json.dumps(result)
    assert client.call_count == 2


def test_cutoff_scenes_are_pending_without_spending_render_bound(monkeypatch, tmp_path):
    listing(monkeypatch, "2026-08-21T15:00:00Z", "2026-08-10T15:00:00Z")
    client = Mock()
    collection = client.ImageCollection.return_value
    collection.aggregate_max.return_value.getInfo.return_value = (
        pd.Timestamp("2026-08-15", tz="UTC").timestamp() * 1000
    )
    monkeypatch.setattr(emit, "_ee_client", lambda: client)
    render = Mock(return_value={"status": "unavailable", "message": "fixture render"})
    monkeypatch.setattr(emit, "fetch_chips", render)
    result = d.enrich(screen(*frames()), cache_dir=tmp_path, max_renders=1)
    scene = at(result, 20)["emit"]["scenes"][0]
    assert scene["imagery"]["status"] == "pending"
    assert "15 Aug 2026" in scene["imagery"]["message"]
    assert render.call_count == 1
    assert render.call_args.args[0]["time_utc"] == "2026-08-10T15:00:00Z"


def test_card_link_matches_unit_and_overlapping_spell():
    n, p, c = history_fixture(days=63, outage=55)
    cards = f.screen(n, p, c)
    result = d.screen(n, p, c, cards=cards)
    linked = [r for r in night_rows(result) if r["night"] >= "2026-09-25"]
    assert linked
    assert all(r["card_id"] == cards["candidates"][0]["id"] for r in linked)
    assert all(
        r["card_id"] is None for r in night_rows(result) if r["night"] < "2026-09-25"
    )


def test_json_round_trip_no_nan_and_tiny_misses_keep_sort_precision(monkeypatch):
    curve = [
        {
            "lower_mw": 0,
            "upper_mw": None,
            "trials": 10000,
            "detections": 9999,
            "probability": 0.9999,
        }
    ]
    monkeypatch.setattr(f, "detection_curve", lambda p: curve)
    result = screen(*frames())
    assert 0 < at(result, 20)["expected_misses"]["miss_all"] < 1e-6
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_empty_selection_has_finite_counts_and_no_listings(monkeypatch, tmp_path):
    payload = screen(*frames(dark=()))
    cmr = Mock(side_effect=AssertionError("no selected sources"))
    monkeypatch.setattr(emit, "cmr_granules", cmr)
    result = d.enrich(payload, cache_dir=tmp_path)
    assert night_rows(result) == [] and result["sources"] == {}
    assert all(count == 0 for count in result["counts"].values())
    assert not cmr.called
    json.dumps(result, allow_nan=False)


def test_source_pass_flags_and_copy_only_positive_brackets():
    n, p, c = frames(dark=(20,))
    prepared_p = f.prepare_passes(n, p)
    prepared_c = f.prepare_copies(c, n.night.min(), n.night.max())
    unit = f.source_units(prepared_c, "2026-08-30")[0]
    _, cp, obs = f.unit_history(unit, prepared_p, prepared_c, sorted(n.night))
    # Wrong assigned-site flags cannot remove the source-positive evidence.
    obs["detected"] = False
    records = d.pass_records(obs[obs.granule != "g19_0"], cp)
    positive = next(r for r in records if r["time_utc"] == "2026-08-21T00:00:00Z")
    assert positive["detected"] and not positive["clear"]
    assert next(r for r in records if r["time_utc"] == "2026-08-21T01:00:00Z")[
        "detected"
    ]
    assert not next(r for r in records if r["time_utc"] == "2026-08-22T00:00:00Z")[
        "detected"
    ]


def test_build_writes_lazy_json_and_monthly_update_rebuilds_it(monkeypatch, tmp_path):
    n, p, c = frames()
    monkeypatch.setattr(b, "find_segments", lambda *a: [(tmp_path, "fixture")])
    monkeypatch.setattr(b, "load_segments", lambda *a: (n, p))
    monkeypatch.setattr(b, "load_copies", lambda *a: c)
    monkeypatch.setattr(
        b, "build_data", lambda *a: ({"nights": sorted(n.night), "sites": []}, {})
    )
    monkeypatch.setattr(b, "load_basemap", lambda: "{}")
    monkeypatch.setattr(b, "explainer_html", lambda **k: "<h3>fixture</h3>")
    monkeypatch.setattr(emit, "enrich", lambda payload, **k: payload)
    listing(monkeypatch)  # network-free, real dark-night enrichment
    monkeypatch.setattr(update, "last_processed", lambda *a: date(2026, 9, 14))
    site = update.main(
        base=tmp_path, site_dir=tmp_path / "site", today=date(2026, 9, 17)
    )
    first = json.loads((site / "darknights.json").read_text())
    assert first["counts"]["dark_nights"] == 1
    assert len(first["runs"]) == 1 and "passes" not in first["sources"]["1"]
    html = (site / "index.html").read_text()
    assert 'fetch("darknights.json")' in html
    assert '"burning_rate":' not in html
    assert 'id="ex-toc"' in html and "tocUpdate" in html
    # No new nights due; monthly refresh still writes a fresh JSON listing.
    (site / "darknights.json").write_text("stale")
    update.main(base=tmp_path, site_dir=site, today=date(2026, 9, 17))
    assert json.loads((site / "darknights.json").read_text()) == first


def test_main_loads_segments_and_runs_offline(monkeypatch, tmp_path):
    n, p, c = frames()
    monkeypatch.setattr(b, "find_segments", lambda: ["fixture"])
    monkeypatch.setattr(b, "load_segments", lambda segments: (n, p))
    monkeypatch.setattr(b, "load_copies", lambda segments: c)
    session = Mock()
    session.get.side_effect = OSError("offline")
    result = d.main(session=session)
    assert result["counts"]["dark_nights"] == 1
    assert result["counts"]["sources"] == 1


@pytest.mark.parametrize(
    "column,zone",
    [
        (639, "swath edge"),
        (640, "mid-swath"),
        (1007, "mid-swath"),
        (1008, "near nadir"),
        (2191, "near nadir"),
        (2192, "mid-swath"),
        (2559, "mid-swath"),
        (2560, "swath edge"),
        (None, "unknown"),
    ],
)
def test_view_zone_boundaries(column, zone):
    assert d._view_zone(column) == zone


def test_leading_twenty_night_run_has_no_earlier_lit_night():
    payload = screen(*frames(days=50, dark=range(20)))
    assert payload["runs"] == []
    assert payload["counts"]["dark_nights"] == 0


@pytest.mark.parametrize("days", [45, 60])
def test_long_run_keeps_only_nights_passing_rate_and_earlier_lit(days):
    payload = screen(*frames(days=days, dark=range(20, 40)))
    # Early run nights have 14 earlier lit nights but 14/28 is not > 0.5.
    # Later nights lose the earlier lit night from their window.
    assert payload["runs"] == []
    payload = screen(*frames(days=60, dark=range(20, 30)))
    run = payload["runs"][0]
    assert len(run["nights"]) == 10
    for night in run["nights"]:
        assert night["burning_rate"] == {"lit": 19, "qualified": 28}
        assert night["passes"]
    assert run["bracket"]["last_lit"] is not None
    # In a partial window, a 12-night ongoing run retains just its first 3.
    payload = screen(*frames(days=32, dark=range(20, 32)))
    run = payload["runs"][0]
    assert run["clear_dark_nights"] == 12
    assert [n["night"] for n in run["nights"]] == [
        "2026-08-21",
        "2026-08-22",
        "2026-08-23",
    ]
    assert [n["burning_rate"] for n in run["nights"]] == [
        {"lit": 14, "qualified": 25},
        {"lit": 13, "qualified": 24},
        {"lit": 12, "qualified": 23},
    ]
    assert run["bracket"]["recovery"] is None


def test_review_order_prefers_high_rate_over_low_rate_high_heat():
    def night(lit, heat, p):
        miss = (1 - p) ** 2
        return {
            "night": "2026-08-21",
            "recent_heat_mw": heat,
            "score": d.wilson_lower_95(lit, 28) * (1 - miss),
        }

    low = night(15, 100, 0.99)
    high = night(27, 3, 0.9)
    assert low["score"] < high["score"]
    assert sorted([low, high], key=d.default_key) == [high, low]
    assert d.wilson_lower_95(0, 0) == 0
    assert d.best_night({"nights": [low, high]}) == high


def test_compact_payload_size_states_trim_and_shared_scenes(monkeypatch, tmp_path):
    payload = screen(
        *frames(days=150, dark=(48, 50, 51, 52, 100), cloudy=(48,), missing_heat=(49,))
    )
    source = payload["sources"]["1"]
    assert "passes" not in source
    assert [s["start"] for s in source["nightly_series"]] == [
        "2026-09-06",
        "2026-10-26",
    ]
    assert [len(s["heat"]) for s in source["nightly_series"]] == [31, 29]
    assert set("".join(s["state"] for s in source["nightly_series"])) == set("LHDU")
    listing(monkeypatch, "2026-09-21T15:00:00Z")
    folder = tmp_path / "cached"
    folder.mkdir()
    for name in ("ch4.png", "rgb.png"):
        (folder / name).write_bytes(b"cached png")
    monkeypatch.setattr(
        emit,
        "cached_chips",
        lambda *a: {
            "status": "available",
            "source_coverage": "covered",
            "cache_folder": str(folder),
            "caption": emit.CAPTION,
        },
    )
    result = d.enrich(payload, site_dir=tmp_path / "site", cache_dir=tmp_path)
    raw = json.dumps(result, separators=(",", ":"), allow_nan=False)
    assert len(raw.encode()) < 5000 + 1800 * result["counts"]["runs"]
    assert "passes" not in result["sources"]["1"]
    assert "source_passes" not in raw
    assert len(result["sources"]["1"]["scenes"]) == 1
    assert raw.count(emit.CAPTION) == 1
    assert raw.count('"unit":') == 1
    assert raw.count('"time_utc":') == 1  # one scene record, pass arrays have no keys
    assert all(len(ref) == 3 for r in result["runs"] for ref in r["emit_scenes"])


def test_comparisons_exclude_windows_of_every_listed_run(monkeypatch, tmp_path):
    listing(
        monkeypatch,
        "2026-08-10T15:00:00Z",
        "2026-08-19T15:00:00Z",
        "2026-08-20T15:00:00Z",
        "2026-08-25T15:00:00Z",
        "2026-08-29T15:00:00Z",
        "2026-09-01T15:00:00Z",
    )
    monkeypatch.setattr(
        emit,
        "cached_chips",
        lambda *a: {"status": "available", "source_coverage": "covered"},
    )
    result = d.enrich(screen(*frames(dark=(20, 30))), cache_dir=tmp_path)
    source = result["sources"]["1"]
    nearby = {sid for r in result["runs"] for sid, _, _ in r["emit_scenes"]}
    assert not nearby.intersection(source["comparison_scenes"])
    assert source["comparison_scenes"]
    for scene in controls(result):
        assert all(
            not (a <= d._instant(scene["time_utc"]) < b)
            for a, b in (d.run_window(r, 0) for r in result["runs"])
        )


def fake_renderer(monkeypatch, tmp_path, fail_first=False):
    """Write real successful caches and keep one persistent render failure."""
    listing(
        monkeypatch,
        "2026-08-21T15:00:00Z",
        "2026-08-22T15:00:00Z",
        "2026-08-23T15:00:00Z",
    )
    client = Mock()
    collection = client.ImageCollection.return_value
    collection.aggregate_max.return_value.getInfo.return_value = (
        pd.Timestamp("2026-08-31", tz="UTC").timestamp() * 1000
    )
    monkeypatch.setattr(emit, "_ee_client", lambda: client)
    attempted = []

    def render(scene, unit, cache_dir, **kwargs):
        attempted.append(scene["scene_id"])
        if fail_first and scene["time_utc"] == "2026-08-21T15:00:00Z":
            return {"status": "unavailable", "message": "fixture failure"}
        folder = emit.chip_folder(scene, unit, cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metadata.json").write_text('{"status":"available"}')
        from PIL import Image

        for name in ("ch4.png", "rgb.png"):
            Image.new("RGB", (512, 512), "#00131f").save(folder / name)
        return {
            "status": "available",
            "source_coverage": "covered",
            "cache_folder": str(folder),
        }

    monkeypatch.setattr(emit, "fetch_chips", render)
    return attempted


@pytest.mark.parametrize("fail_first", [False, True])
def test_render_queue_advances_across_builds_and_failure_backoff(
    monkeypatch, tmp_path, fail_first
):
    attempted = fake_renderer(monkeypatch, tmp_path, fail_first)
    now = d._instant("2026-09-15T00:00:00Z")
    for count in range(1, 4):
        result = d.enrich(
            screen(*frames(dark=(20, 21, 22))),
            cache_dir=tmp_path,
            site_dir=tmp_path / "site",
            max_renders=1,
            now=now,
        )
        assert len(attempted) == count
        assert result["rendering"]["new_renders"] == 1
    assert len(set(attempted)) == 3
    state = json.loads((tmp_path / "darknight_attempts.json").read_text())
    assert len(state) == 4  # three scenes plus collection horizon
    d.enrich(
        screen(*frames(dark=(20, 21, 22))), cache_dir=tmp_path, max_renders=1, now=now
    )
    assert len(attempted) == 3
    d.enrich(
        screen(*frames(dark=(20, 21, 22))),
        cache_dir=tmp_path,
        max_renders=1,
        now=now + timedelta(days=1),
    )
    assert len(attempted) == (4 if fail_first else 3)
    if fail_first:
        first_key = emit.chip_folder(
            {"scene_id": attempted[0]}, {"lat": 0.0, "lon": 0.0}, tmp_path
        ).name
        state = json.loads((tmp_path / "darknight_attempts.json").read_text())
        assert state[first_key]["attempts"] == 2
        assert d._instant(state[first_key]["retry_after"]) == now + timedelta(days=3)


def test_zero_render_budget_keeps_cutoff_scenes_pending(monkeypatch, tmp_path):
    attempted = fake_renderer(monkeypatch, tmp_path)
    listing(monkeypatch, "2026-09-02T15:00:00Z")
    payload = screen(*frames(dark=(30, 31, 32)))
    result = d.enrich(payload, max_renders=0, cache_dir=tmp_path)
    scene = next(iter(result["sources"]["1"]["scenes"].values()))
    assert scene["imagery"]["status"] == "pending"
    assert not attempted
    monkeypatch.setattr(
        emit, "_ee_client", Mock(side_effect=AssertionError("use saved cutoff"))
    )
    result = d.enrich(
        screen(*frames(dark=(30, 31, 32))), max_renders=-1, cache_dir=tmp_path
    )
    assert (
        next(iter(result["sources"]["1"]["scenes"].values()))["imagery"]["status"]
        == "pending"
    )


def test_closed_month_listings_reuse_final_cache_on_rebuild(tmp_path):
    session = Mock()
    session.get.return_value.json.return_value = {"feed": {"entry": []}}
    now = d._instant("2026-11-09T00:00:00Z")
    for expected in (2, 4):
        result = d.enrich(
            screen(*frames(days=100)),
            imagery=False,
            cache_dir=tmp_path,
            session=session,
            now=now,
        )
        assert result["counts"]["sources"] == 1
        assert session.get.call_count == expected
    # Aug/Sep are final; October/November share one query per product.
    assert {
        c.kwargs["params"]["temporal"] for c in session.get.call_args_list[-2:]
    } == {emit.utc_window("2026-10-01", "2026-11-30", 0)}
    assert list(d.calendar_months("2026-08-09", "2026-11-08")) == [
        ("2026-08-01", "2026-08-31"),
        ("2026-09-01", "2026-09-30"),
        ("2026-10-01", "2026-10-31"),
        ("2026-11-01", "2026-11-30"),
    ]


def test_nearby_overlapping_dark_runs_link_only_matching_dark_nights():
    frames_by_site = []
    for sid, lon, dark in [
        (1, 0, (20, 21, 30)),
        (2, 0.01, (20, 21, 30)),
        (3, 0.02, (20, 21, 30)),
        (4, 0.05, (20, 21, 30)),
    ]:
        ns, ps, cs = frames(dark=dark)
        ns["site_id"] = sid
        ns["lon"] = lon
        ps["site_id"] = sid
        cs["assigned_site"] = sid
        cs["group_id"] = sid
        cs["centroid_lon"] = lon
        frames_by_site.append((ns, ps, cs))
    result = screen(
        *(
            pd.concat([parts[i] for parts in frames_by_site], ignore_index=True)
            for i in range(3)
        )
    )
    assert len(result["sources"]) == 4
    by_id = {r["id"]: r for r in result["runs"]}
    for run in result["runs"]:
        assert all(by_id[other]["start"] == run["start"] for other in run["neighbours"])
        if run["source_id"] == "4":
            assert run["neighbours"] == []
        else:
            assert run["neighbours"]
            assert all(
                run["id"] in by_id[other]["neighbours"] for other in run["neighbours"]
            )


def test_new_scene_slots_are_reserved_while_a_failed_scene_is_due(
    monkeypatch, tmp_path
):
    attempted = fake_renderer(monkeypatch, tmp_path, fail_first=True)
    now = d._instant("2026-09-15T00:00:00Z")
    d.enrich(
        screen(*frames(dark=(20, 21, 22))), cache_dir=tmp_path, max_renders=1, now=now
    )
    first = attempted[0]
    d.enrich(
        screen(*frames(dark=(20, 21, 22))),
        cache_dir=tmp_path,
        max_renders=2,
        now=now + timedelta(days=1),
    )
    assert len(attempted) == 3
    assert attempted[1] != first  # the new-scene slot is protected
    assert attempted[2] == first  # a due failure also gets a retry slot
