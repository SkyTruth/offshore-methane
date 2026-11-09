"""Regressions for the consolidated round-one review findings."""

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock

import pandas as pd
import pytest

import build_viewer as b
import emit
import flareouts as f
import pipeline
import update
from test_flareouts import candidate, fake_response, fixture_states, history_fixture


def test_zero_comparisons_caps_a_very_long_spell_at_mid():
    c = f.screen(*history_fixture(days=110))["candidates"][0]
    assert c["score"]["budget_multiple"] < 1
    assert c["tier"] == "Mid"
    assert c["context"]["label"] == "no comparison flare within 10 km"
    assert not c["context"]["isolation_resolved"]


def test_cloud_only_comparison_heat_cannot_dilute_a_measured_loss():
    days = pd.date_range("2026-08-01", periods=40).strftime("%Y-%m-%d")
    h = pd.DataFrame(
        dict(lit=True, qualified=True, heat_mw=12.0, clear_passes=1, clear_groups=1),
        index=days,
    )
    h.loc[days[15:18], "heat_mw"] = 0.5
    h.loc[days[18:], "heat_mw"] = float("nan")
    h.loc[days[18:], "qualified"] = False
    units = [
        dict(
            id=str(i),
            lat=24.0,
            lon=53.0 + 0.07 * i,
            baseline_group_count=30,
            site_ids=[i],
        )
        for i in (0, 1)
    ]
    ctx = f.shared_context(
        units[0], dict(start=days[15], end=days[-1]), units, {"1": h}
    )
    donor = ctx["neighbours"][0]
    assert donor["measured_nights"] == 3
    assert donor["shared_heat_loss"]
    assert ctx["label"] == "shared heat loss"
    assert not ctx["isolation_resolved"]
    # Even non-losing measured nights cannot resolve 22 unknown nights.
    h.loc[days[15:18], "heat_mw"] = 12.0
    ctx = f.shared_context(
        units[0], dict(start=days[15], end=days[-1]), units, {"1": h}
    )
    assert ctx["label"] == "comparison heat unresolved"
    assert not ctx["isolation_resolved"]


@pytest.mark.parametrize("days", [88, 180])
def test_six_night_927_spell_is_mid_at_58_and_150_evaluation_nights(days):
    n, p, c = history_fixture(days=days, outage=days - 6)
    # Exactly 12 clear, lit baseline nights; cloudy positive nights reset only.
    start = pd.Timestamp("2026-08-01") + pd.Timedelta(days=days - 6 - 29)
    end = start + pd.Timedelta(days=17)
    cloudy = pd.to_datetime(p.time_utc).between(
        start + pd.Timedelta(days=1), end, inclusive="left"
    )
    p.loc[cloudy, "cloud_mask"] = 3
    cp_cloudy = pd.to_datetime(c.time_utc).between(
        start + pd.Timedelta(days=1), end, inclusive="left"
    )
    c.loc[cp_cloudy, "cloud_mask_peak"] = 3
    result = f.screen(n, p, c)
    card = result["candidates"][0]
    assert card["tier"] == "Mid"
    assert card["dark_qualified_nights"] == 6
    assert card["score"]["evaluation_nights"] == 30
    assert "independent_adjusted" not in card["score"]


def test_later_source_is_admitted_after_appended_month():
    n, p, c = history_fixture(days=100, outage=75)
    cutoff = "2026-08-31T00:00:00"
    c = c[c.time_utc >= cutoff]
    p.loc[p.time_utc < cutoff, "detected"] = False
    first = f.screen(
        n[n.night < "2026-10-01"],
        p[p.time_utc < "2026-10-02"],
        c[c.time_utc < "2026-10-02"],
    )
    assert not first["candidates"]
    full = f.screen(n, p, c)
    assert full["candidates"][0]["tier"] == "Mid"
    assert full["candidates"][0]["baseline"]["lit_fraction"] > 0.8
    assert first["calibration"]["eligible_units_by_night"]["2026-09-01"] == 0
    assert full["calibration"]["eligible_units_by_night"]["2026-10-11"] == 1


def test_source_localisation_19_of_29_does_not_inherit_site_27_of_29():
    n, p, c = history_fixture(days=88, outage=82)
    baseline = pd.date_range("2026-09-23", periods=29).strftime("%Y-%m-%d")
    # Assigned detections remain positive, but ten source nights are outside
    # the frozen aperture; alter later nights to preserve the learned centre.
    local_night = (pd.to_datetime(c.time_utc) - pd.Timedelta(hours=9)).dt.strftime(
        "%Y-%m-%d"
    )
    lit_nights = sorted(set(local_night[local_night.isin(baseline)]))
    moved = local_night.isin(lit_nights[:8])
    c.loc[moved, "centroid_lat"] += 0.04
    result = f.screen(n, p, c)
    assert not any(card["status"] == "ongoing" for card in result["candidates"])
    pc = f.prepare_copies(c, n.night.min(), n.night.max())
    unit = f.source_units(pc, "2026-08-30")[0]
    h, _, _ = f.unit_history(unit, f.prepare_passes(n, p), pc, sorted(n.night.unique()))
    baseline_stats = f.baseline_stats(h, "2026-10-21")
    assert baseline_stats["lit_qualified_nights"] == 19
    assert not f.eligible(baseline_stats)
    assert (
        p[p.time_utc.between("2026-09-24", "2026-10-23")]
        .groupby(p.time_utc.str[:10])
        .detected.max()
        .sum()
        == 27
    )


def test_4582_shape_with_measured_nonlosing_comparison_still_fails_high():
    # 23 source units, 58 evaluation nights, 14 dark nights, 3 baseline misses.
    n, p, c = history_fixture(days=88, outage=88)
    nn, pp, cc = [], [], []
    for sid in range(23):
        ni, pi, ci = n.copy(), p.copy(), c.copy()
        ni["site_id"] = sid + 1
        pi["site_id"] = sid + 1
        ci["assigned_site"] = sid + 1
        lon = 53.0 + sid * 0.07
        if sid > 1:
            lon = 53.0 + sid * 2
        ni["lon"] = lon
        ci["centroid_lon"] = lon
        ci["group_id"] = sid + 1
        if sid == 0:
            nt = (
                pd.to_datetime(ci.time_utc) + pd.Timedelta(hours=53 / 15 - 12)
            ).dt.strftime("%Y-%m-%d")
            missing = nt >= "2026-10-14"
            baseline_miss = nt.isin(["2026-09-18", "2026-09-25", "2026-10-02"])
            ci = ci[~(missing | baseline_miss)]
            pt = (
                pd.to_datetime(pi.time_utc) + pd.Timedelta(hours=53 / 15 - 12)
            ).dt.strftime("%Y-%m-%d")
            pi.loc[
                (pt >= "2026-10-14")
                | pt.isin(["2026-09-18", "2026-09-25", "2026-10-02"]),
                "detected",
            ] = False
        nn.append(ni)
        pp.append(pi)
        cc.append(ci)
    data = f.screen(*(pd.concat(x, ignore_index=True) for x in (nn, pp, cc)))
    target = data["candidates"][0]
    assert target["dark_qualified_nights"] == 14
    assert target["context"]["isolation_resolved"]
    assert target["score"]["budget_multiple"] > 1
    assert target["tier"] == "Mid"


def test_nearby_assignment_fragments_produce_one_card():
    n, p, c = history_fixture()
    copies = [c]
    for sid, offset in [(3, 0.007), (4, 0.010), (5, 0.012)]:
        part = c.iloc[::7].copy()
        part["assigned_site"] = sid
        part["group_id"] = sid
        part["centroid_lat"] += offset
        copies.append(part)
    cards = f.screen(n, p, pd.concat(copies, ignore_index=True))["candidates"]
    assert len(cards) == 1
    assert cards[0]["unit"]["site_ids"] == [1]


def test_dated_50_to_8_mw_turndown_is_scored_from_reduced_heat():
    n, p, c = history_fixture(days=80, outage=55)
    c["group_rh_mw"] = 50.0
    c.loc[c.time_utc >= "2026-09-06", "group_rh_mw"] = 8.0
    data = f.screen(n, p, c)
    card = data["candidates"][0]
    assert card["tier"] == "Mid"
    assert card["pre_gap_heat_mw"] == 8.0
    assert card["pre_spell_state"]["state"] == "reduced heat"
    assert card["pre_spell_state"]["start"] < card["start"]


def test_short_gaps_merge_and_recovery_has_own_heat_row():
    n, p, c = history_fixture()
    recovery = c.iloc[0].copy()
    recovery["time_utc"] = "2026-10-19T00:00:00"
    recovery["granule"] = "recovery"
    recovery["group_rh_mw"] = 0.19
    c.loc[len(c)] = recovery
    states = fixture_states(n, p, c)
    dark = [s for s in states if s["state"] == "unresolved darkness"]
    assert len(dark) == 1
    assert dark[0]["measured_nights"] == 0
    assert dark[0]["end"] == "2026-10-17"
    assert states[-1]["start"] == "2026-10-18"
    assert states[-1]["measured_nights"] == 1
    assert states[-1]["median_mw"] == 0.19


def test_empty_copy_table_replaces_old_positive_by_manifest(tmp_path):
    segments = []
    for i in (0, 1):
        folder = tmp_path / str(i)
        folder.mkdir()
        tag = str(i)
        frame = pd.DataFrame(
            dict(
                satellite=["SNPP"],
                granule=["g"],
                group_id=[1],
                peak_row=[1],
                peak_col=[1],
            )
        )
        frame.iloc[: 1 - i].to_csv(folder / f"copies_{tag}.csv", index=False)
        pd.DataFrame(
            dict(satellite=["SNPP"], granule=["g"], complete=[True], n_sites=[1])
        ).to_csv(folder / f"jobs_{tag}.csv", index=False)
        segments.append((folder, tag))
    assert b.load_copies(segments).empty


@pytest.mark.parametrize("lon", [-90.0, 90.0])
def test_emit_morning_afternoon_boundary_dates_use_viirs_brackets(lon):
    c = candidate()
    c["unit"]["lon"] = lon

    # Last lit pass: start morning, first dark: next morning; recovery: end+1.
    def utc(local):
        return (
            datetime.fromisoformat(local).replace(tzinfo=timezone.utc)
            - timedelta(hours=lon / 15)
        ).isoformat()

    c["viirs_times"] = dict(
        last_lit=utc("2026-10-30T02:00"),
        first_dark=utc("2026-10-31T02:00"),
        last_dark=utc("2026-11-05T02:00"),
        recovery=utc("2026-11-06T02:00"),
    )
    for local, expected in [
        ("2026-10-30T09:00", "boundary"),
        ("2026-10-30T16:00", "boundary"),
        ("2026-10-31T09:00", "during"),
        ("2026-11-05T09:00", "boundary"),
        ("2026-11-05T16:00", "boundary"),
        ("2026-11-06T09:00", "after"),
    ]:
        assert emit.scene_phase(datetime.fromisoformat(utc(local)), c) == expected
    bounds = emit.utc_window("2026-10-30", "2026-11-04", lon)
    assert bounds.split(",")[0] == utc("2026-10-30T12:00").replace("+00:00", "Z")
    assert bounds.split(",")[1] == utc("2026-11-05T12:00").replace("+00:00", "Z")


def test_cmr_late_update_window_not_final_at_14_days(tmp_path):
    session = Mock()
    session.get.side_effect = [fake_response([]), fake_response([dict(id="late")])]
    args = ("EMITL1BRAD", 1, 2, "2026-10-01", "2026-10-15")
    assert (
        emit.cmr_granules(
            *args, cache_dir=tmp_path, today=date(2026, 10, 20), session=session
        )
        == []
    )
    assert emit.cmr_granules(
        *args, cache_dir=tmp_path, today=date(2026, 10, 29), session=session
    ) == [dict(id="late")]
    assert session.get.call_count == 2


def test_offline_listing_and_chip_cache_survive_without_ee(tmp_path, monkeypatch):
    entry = dict(
        producer_granule_id="EMIT_L1B_RAD_002_20261025T154451",
        time_start="2026-10-25T15:44:51Z",
    )
    session = Mock()
    session.get.return_value = fake_response([entry])
    data = dict(as_of="2026-11-04", candidates=[candidate()])
    emit.enrich(data, cache_dir=tmp_path, session=session, imagery=False)
    scene = data["candidates"][0]["emit"]["scenes"][0]
    folder = emit.chip_folder(scene, candidate()["unit"], tmp_path)
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps(
            dict(status="available", source_coverage="covered", caption=emit.CAPTION)
        )
    )
    from PIL import Image

    for name in ("ch4.png", "rgb.png"):
        Image.new("RGB", (512, 512), "#00131f").save(folder / name)
    client = Mock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(emit, "_ee_client", client)
    session.get.side_effect = RuntimeError("offline")
    emit.enrich(data, cache_dir=tmp_path, site_dir=tmp_path / "site", session=session)
    assert not client.called
    assert data["candidates"][0]["emit"]["status"] == "unavailable"
    image = data["candidates"][0]["emit"]["scenes"][0]["imagery"]
    assert image["status"] == "available"
    assert (tmp_path / "site" / image["ch4"]).read_bytes() == (
        folder / "ch4.png"
    ).read_bytes()


def write_segment_stub(base, tag, summary=None, phase="done"):
    folder = base / f"v5_update_{tag}"
    folder.mkdir(exist_ok=True)
    pd.DataFrame(dict(night=[tag[11:]])).to_csv(
        folder / f"site_nights_{tag}.csv", index=False
    )
    (folder / f"overpasses_{tag}.csv").write_text("site_id\n")
    (folder / "status.json").write_text(json.dumps(dict(phase=phase)))
    if summary is not None:
        (folder / f"run_summary_{tag}.json").write_text(json.dumps(summary))
    return folder


def test_shrinking_manifest_is_rejected_before_build_or_publication(
    tmp_path, monkeypatch
):
    tag = "2026-11-01_2026-11-06"
    folder = write_segment_stub(tmp_path, tag, dict(expected=2, complete=1))

    def shrink(**kw):
        (folder / f"run_summary_{tag}.json").write_text(
            json.dumps(dict(expected=1, complete=1))
        )

    monkeypatch.setattr(update.backfill, "main", shrink)
    build = Mock()
    publish = Mock()
    monkeypatch.setattr(b, "main", build)
    monkeypatch.setattr(b, "publish_pages", publish)
    with pytest.raises(RuntimeError, match="vanished"):
        update.main(
            base=tmp_path, today=date(2026, 11, 9), publish=True, accept_missing=True
        )
    assert not build.called and not publish.called
    # A second invocation cannot certify the now-shrunk summary either.
    with pytest.raises(RuntimeError, match="vanished"):
        update.main(base=tmp_path, today=date(2026, 11, 9), publish=True)
    assert not build.called and not publish.called


def test_expired_resume_checks_retention_before_backfill(tmp_path, monkeypatch):
    write_segment_stub(
        tmp_path, "2026-07-01_2026-07-30", dict(expected=100, complete=50)
    )
    backfill = Mock()
    monkeypatch.setattr(update.backfill, "main", backfill)
    with pytest.warns(UserWarning, match="91 days"):
        with pytest.raises(ValueError, match="retention"):
            update.main(base=tmp_path, today=date(2026, 11, 9), accept_missing=True)
    assert not backfill.called


def test_expired_but_fully_cached_jobs_can_resume_validation(tmp_path, monkeypatch):
    tag = "2026-07-01_2026-07-30"
    folder = write_segment_stub(
        tmp_path, tag, dict(expected=1, complete=1), phase="validate"
    )
    job = dict(satellite="SNPP", gid="cached", start="2026-07-02 00:00:00", sites=[])
    (folder / f"expected_jobs_{tag}.json").write_text(json.dumps([job]))
    cache = folder / "overpasses_v5" / "SNPP"
    cache.mkdir(parents=True)
    for name in ("cached.csv", "cached_copies.csv"):
        (cache / name).write_text("header\n")
    backfill = Mock()
    monkeypatch.setattr(update.backfill, "main", backfill)
    update.process_segment(folder, tag[:10], tag[11:], date(2026, 11, 9), 1, False)
    assert backfill.called


def test_permanent_missing_granule_can_be_acknowledged_and_next_month_runs(
    tmp_path, monkeypatch
):
    tag = "2026-11-01_2026-11-06"
    summary = dict(expected=2, complete=1, missing=[["SNPP", "bad"]])
    folder = write_segment_stub(tmp_path, tag, summary)
    pd.DataFrame(
        dict(site_id=[1], night=["2026-11-05"], satellite=["SNPP"], granule=["bad"])
    ).to_csv(folder / f"missing_observations_{tag}.csv", index=False)
    calls = []

    def run(**kw):
        calls.append((kw["start"], kw["end"]))
        if kw["out_dir"] != folder:
            write_segment_stub(
                tmp_path, f"{kw['start']}_{kw['end']}", dict(expected=1, complete=1)
            )

    monkeypatch.setattr(update.backfill, "main", run)
    build = Mock(return_value=tmp_path / "site")
    monkeypatch.setattr(b, "main", build)
    with pytest.warns(UserWarning, match="Accepted 1"):
        update.main(base=tmp_path, today=date(2026, 11, 9), accept_missing=True)
    update.main(base=tmp_path, today=date(2026, 12, 9))
    assert calls == [("2026-11-01", "2026-11-06"), ("2026-11-07", "2026-12-06")]
    assert build.call_count == 2


def test_persistent_expected_jobs_survive_noaa_listing_shrink(tmp_path):
    tag = "2026-11-01_2026-11-06"
    jobs = [
        dict(
            satellite="SNPP",
            gid=gid,
            start=datetime(2026, 11, 2),
            keys=dict(M10="public"),
            sites=[[1, 24, 53]],
        )
        for gid in ("cached", "bad")
    ]
    pipeline.persistent_jobs(jobs, tmp_path, tag)
    resumed = pipeline.persistent_jobs(jobs[:1], tmp_path, tag)
    assert [j["gid"] for j in resumed] == ["cached", "bad"]
    assert resumed[1]["keys"] == dict(M10="public")
    missing = pipeline.missing_observations(
        resumed,
        tmp_path / "cache",
        pd.DataFrame(dict(site_id=[1])),
        date(2026, 11, 1),
        date(2026, 11, 6),
    )
    assert set(missing.night) == {"2026-11-01"}


def test_three_segments_through_loaders_and_real_site_build(tmp_path, monkeypatch):
    n, p, c = history_fixture(days=90, outage=65)
    p["temp_k"] = 1500.0
    p["bytes_granule"] = 100
    c["peak_row"] = 1
    c["peak_col"] = 1
    sites = n.drop_duplicates("site_id")[["site_id", "lat", "lon"]].copy()
    sites["n_structures"] = 1
    sites["structure_ids"] = "fixture"
    sites.to_csv(tmp_path / "gfw_oil_sites.csv", index=False)
    segments = []
    for i in range(3):
        days = sorted(n.night.unique())[i * 30 : (i + 1) * 30]
        tag = f"{days[0]}_{days[-1]}"
        folder = tmp_path / f"v5_fixture_{i}"
        folder.mkdir()
        pi = p[
            (pd.to_datetime(p.time_utc) - pd.Timedelta(hours=9))
            .dt.strftime("%Y-%m-%d")
            .isin(days)
        ]
        ci = c[
            (pd.to_datetime(c.time_utc) - pd.Timedelta(hours=9))
            .dt.strftime("%Y-%m-%d")
            .isin(days)
        ]
        nightly = pipeline.site_nights(
            pi, sites, date.fromisoformat(days[0]), date.fromisoformat(days[-1])
        )
        nightly.to_csv(folder / f"site_nights_{tag}.csv", index=False)
        pi.to_csv(folder / f"overpasses_{tag}.csv", index=False)
        ci.to_csv(folder / f"copies_{tag}.csv", index=False)
        pi[["satellite", "granule"]].drop_duplicates().to_csv(
            folder / f"jobs_{tag}.csv", index=False
        )
        segments.append((folder, tag))
    loaded_n, loaded_p = b.load_segments(b.find_segments(tmp_path))
    loaded_c = b.load_copies(segments)
    assert len(loaded_n) == 90 and len(loaded_p) == 270
    assert len(loaded_c) == len(c)
    actual_build_data = b.build_data
    monkeypatch.setattr(b, "build_data", lambda n, p: actual_build_data(n, p, tmp_path))
    myc = pd.DataFrame(
        dict(
            flare_id=["fixture"],
            lat=[24.0],
            lon=[53.0],
            regime=["persistent"],
            bcm=[0.1],
            upstream_class=["oil"],
            country_iso=["XX"],
            n_det=[5],
            n_clear=[10],
        )
    )
    monkeypatch.setattr(b.validate_myc25, "load_myc", lambda year: myc)
    monkeypatch.setattr(b, "load_basemap", lambda: '{"type":"Topology","objects":{}}')
    monkeypatch.setattr(emit, "enrich", lambda payload, **kw: payload)
    import darknights

    monkeypatch.setattr(darknights, "enrich", lambda payload, **kw: payload)
    site = b.main(segments=segments, site_dir=tmp_path / "site")
    payload = json.loads((site / "flareouts.json").read_text())
    assert payload["as_of"] == "2026-10-29"
    assert payload["candidates"][0]["dark_qualified_nights"] == 25
    assert "heat_state_records" not in payload
    assert (site / "passes" / "1.json").exists()
    assert "__FLAREOUTS_DATA__" not in (site / "index.html").read_text()
    dark = json.loads((site / "darknights.json").read_text())
    assert dark["as_of"] == "2026-10-29"
    assert dark["counts"]["dark_nights"] > 0


def test_empty_incomplete_pipeline_exports_unknown_nights(tmp_path, monkeypatch):
    sites = pd.DataFrame(dict(site_id=[1], lat=[24.0], lon=[53.0]))
    sites.to_csv(tmp_path / "sites.csv", index=False)
    job = dict(
        satellite="SNPP",
        gid="failed",
        start=datetime(2026, 11, 2),
        keys={},
        sites=[[1, 24, 53]],
    )
    monkeypatch.setattr(pipeline, "build_jobs", lambda *a: [job])
    monkeypatch.setattr(pipeline, "run_jobs", lambda *a: pd.DataFrame())
    result = pipeline.main(
        start="2026-11-01",
        end="2026-11-01",
        sites_csv=tmp_path / "sites.csv",
        out_dir=tmp_path,
    )
    assert result.status.tolist() == ["not_observed"]
    assert result.n_clear.tolist() == [0]
    missing = pd.read_csv(tmp_path / "missing_observations_2026-11-01_2026-11-01.csv")
    assert missing.granule.tolist() == ["failed"]
    # Fully cached expected sets do not depend on today's NOAA listing.
    monkeypatch.setattr(
        pipeline,
        "build_jobs",
        Mock(side_effect=AssertionError("must reuse expected set")),
    )
    pipeline.main(
        start="2026-11-01",
        end="2026-11-01",
        sites_csv=tmp_path / "sites.csv",
        out_dir=tmp_path,
    )


def test_completed_shrunk_summary_cannot_bypass_expected_snapshot(tmp_path):
    tag = "2026-11-01_2026-11-06"
    folder = write_segment_stub(tmp_path, tag, dict(expected=1, complete=1))
    (folder / f"expected_jobs_{tag}.json").write_text(
        json.dumps([dict(satellite="SNPP", gid=g) for g in ("one", "two")])
    )
    with pytest.raises(RuntimeError, match="vanished"):
        update.main(base=tmp_path, today=date(2026, 11, 9))


def test_half_measured_nonlosing_comparison_does_not_prove_isolation():
    days = pd.date_range("2026-08-01", periods=28).strftime("%Y-%m-%d")
    h = pd.DataFrame(
        dict(lit=True, qualified=True, heat_mw=12.0, clear_passes=1, clear_groups=1),
        index=days,
    )
    h.loc[days[21:], "heat_mw"] = float("nan")
    h.loc[days[21:], "qualified"] = False
    units = [
        dict(
            id=str(i),
            lat=24.0,
            lon=53.0 + 0.07 * i,
            baseline_group_count=30,
            site_ids=[i],
        )
        for i in (0, 1)
    ]
    ctx = f.shared_context(
        units[0], dict(start=days[14], end=days[-1]), units, {"1": h}
    )
    assert ctx["neighbours"][0]["measured_nights"] == 7
    assert ctx["label"] == "comparison heat unresolved"
    # One more measured non-losing night rules out a ≥half-spell loss.
    h.loc[days[21], ["heat_mw", "qualified"]] = [12.0, True]
    ctx = f.shared_context(
        units[0], dict(start=days[14], end=days[-1]), units, {"1": h}
    )
    assert ctx["label"] == "isolated"


def test_empty_new_granule_snapshot_also_replaces_stale_exposure(tmp_path):
    segments = []
    for i in (0, 1):
        folder = tmp_path / str(i)
        folder.mkdir()
        tag = str(i)
        pd.DataFrame(
            dict(
                site_id=[1],
                night=["2026-11-01"],
                status=["flaring" if i == 0 else "not_observed"],
            )
        ).to_csv(folder / f"site_nights_{tag}.csv", index=False)
        over = pd.DataFrame(
            dict(
                satellite=["SNPP"],
                granule=["g"],
                site_id=[1],
                time_utc=["2026-11-02T01:00:00"],
            )
        )
        over.iloc[: 1 - i].to_csv(folder / f"overpasses_{tag}.csv", index=False)
        pd.DataFrame(dict(satellite=["SNPP"], granule=["g"])).to_csv(
            folder / f"jobs_{tag}.csv", index=False
        )
        segments.append((folder, tag))
    nights, over = b.load_segments(segments)
    assert nights.status.tolist() == ["not_observed"]
    assert over.empty
